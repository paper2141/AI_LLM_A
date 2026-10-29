"""
도시 정책 RAG 서버 (Google Cloud Run / FastAPI)

파이프라인
  [PDF 업로드] -> /extract : 페이지별 텍스트 추출
  [벡터 DB 생성] -> /ingest : 청크 분할 -> OpenAI 임베딩 -> Supabase(pgvector) 저장 (진행 상황 SSE)
  [질의] -> /query : 질의 확장(Query Augmentation) -> Supabase 검색 -> Gemini 답변 스트리밍 (SSE)

필요한 환경 변수 (코드에 키를 직접 쓰지 않습니다)
  GOOGLE_API_KEY       Google AI(Gemini) API 키
  OPENAI_API_KEY       OpenAI 임베딩용 API 키
  SUPABASE_URL         예) https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY Supabase service_role 키 (서버에서만 사용)
선택
  GEMINI_MODEL         기본 gemini-3.5-flash-lite
  EMBEDDING_MODEL      기본 text-embedding-3-small (1536차원, supabase.sql 과 일치해야 함)
  API_SECRET           설정하면 모든 요청에 X-API-Key 헤더가 일치해야 함
  TOP_K                검색 청크 수 (기본 5)
  CHUNK_SIZE / CHUNK_OVERLAP  청크 크기/겹침 (기본 800 / 120)
"""

import io
import json
import logging
import os
import queue
import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-server")

app = FastAPI(title="Urban Policy RAG Server")

# --------------------------------------------------------------------------
# 프롬프트
# --------------------------------------------------------------------------

# 질의 확장용 시스템 메시지: 대화 맥락을 이용해 대명사를 명확한 명사로 바꾼 한 문장 질문을 만든다.
AUGMENT_SYSTEM = (
    "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
    "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. "
    "질문에 대한 답변은 하지 말고, 변환된 질문 한 문장만 출력하라."
)

# 답변 생성용 시스템 메시지
ANSWER_SYSTEM = (
    "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n"
    "아래 [문서] 에 있는 내용만 근거로 답변하고, 문서에 없는 내용은 추측하지 말고 "
    "'제공된 문서에서 해당 내용을 찾을 수 없습니다'라고 답해라.\n"
    "근거가 된 문서는 답변 끝에 [출처 번호] 형식으로 표기해라.\n\n[문서]\n{context}"
)


# --------------------------------------------------------------------------
# 오류 처리: 어느 단계에서 / 무슨 오류가 / 어떻게 해결할지 를 함께 전달
# --------------------------------------------------------------------------

class StageError(Exception):
    """단계(stage) 정보를 포함한 오류. 클라이언트에 그대로 전달된다."""

    def __init__(self, stage: str, message: str, hint: str = "", status: int = 500):
        super().__init__(message)
        self.stage, self.message, self.hint, self.status = stage, message, hint, status

    def payload(self) -> dict:
        return {"stage": self.stage, "message": self.message, "hint": self.hint}


@contextmanager
def stage(name: str, hint: str = ""):
    """with 블록 안의 예외를 StageError(단계명+예외종류+메시지)로 변환한다."""
    try:
        yield
    except StageError:
        raise
    except Exception as e:  # noqa: BLE001 - 모든 예외를 단계 정보와 함께 보고
        log.exception("[%s] 실패", name)
        raise StageError(name, f"{type(e).__name__}: {e}", hint) from e


def env(name: str, default: Optional[str] = None) -> str:
    """환경 변수를 읽고, 필수인데 없으면 어떤 변수가 빠졌는지 명시한 오류를 낸다."""
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise StageError(
            "설정",
            f"환경 변수 '{name}' 가 설정되지 않았습니다.",
            f"Cloud Run 서비스의 환경 변수(또는 Secret)에 {name} 을 추가한 뒤 다시 배포하세요.",
            status=500,
        )
    return value


def check_api_key(x_api_key: Optional[str] = Header(default=None)):
    """API_SECRET 이 설정된 경우에만 X-API-Key 헤더를 검사한다."""
    secret = os.environ.get("API_SECRET")
    if secret and x_api_key != secret:
        raise StageError(
            "인증", "X-API-Key 헤더가 없거나 일치하지 않습니다.",
            "Apps Script 스크립트 속성 API_SECRET 과 Cloud Run 환경 변수 API_SECRET 값을 동일하게 맞추세요.",
            status=401,
        )


@app.exception_handler(StageError)
async def stage_error_handler(_, exc: StageError):
    return JSONResponse(status_code=exc.status, content={"ok": False, "error": exc.payload()})


# --------------------------------------------------------------------------
# 외부 서비스 클라이언트 (최초 사용 시 생성 -> 환경 변수가 없어도 /health 는 응답 가능)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_supabase():
    return create_client(env("SUPABASE_URL"), env("SUPABASE_SERVICE_KEY"))


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=env("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=env("OPENAI_API_KEY"),
    )


def get_llm(streaming: bool) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=env("GOOGLE_API_KEY"),
        temperature=0.2 if streaming else 0.0,
        streaming=streaming,
    )


# --------------------------------------------------------------------------
# StreamHandler: LLM 이 토큰을 생성할 때마다 큐에 넣어 SSE 로 흘려보낸다
# --------------------------------------------------------------------------

class StreamHandler(BaseCallbackHandler):
    def __init__(self, q: "queue.Queue"):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.q.put({"type": "token", "text": token})


def sse(obj: dict) -> str:
    """SSE 한 건 직렬화 (data: {...}\\n\\n)."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def sse_stream(worker) -> StreamingResponse:
    """worker(q) 를 별도 스레드에서 실행하고, 큐에 쌓이는 이벤트를 SSE 로 내보낸다."""
    q: "queue.Queue" = queue.Queue()

    def run():
        try:
            worker(q)
        except StageError as e:
            q.put({"type": "error", **e.payload()})
        except Exception as e:  # noqa: BLE001
            log.exception("예상하지 못한 오류")
            q.put({"type": "error", "stage": "알 수 없음",
                   "message": f"{type(e).__name__}: {e}", "hint": "Cloud Run 로그를 확인하세요."})
        finally:
            q.put(None)  # 종료 신호

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield sse(item)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# 엔드포인트
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    """연동 상태 확인: 환경 변수 설정 여부, Supabase 연결, 저장된 청크 수. (키 값은 노출하지 않음)"""
    keys = ["GOOGLE_API_KEY", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
    missing = [k for k in keys if not os.environ.get(k)]
    result = {
        "ok": not missing,
        "missing_env": missing,
        "llm_model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "supabase": None,
        "chunk_count": None,
    }
    if not any(k.startswith("SUPABASE") for k in missing):
        try:
            res = get_supabase().table("documents").select("id", count="exact").limit(1).execute()
            result["supabase"] = "connected"
            result["chunk_count"] = res.count
        except Exception as e:  # noqa: BLE001
            result["ok"] = False
            result["supabase"] = f"{type(e).__name__}: {e} (supabase.sql 을 실행했는지 확인하세요)"
    return result


@app.post("/extract", dependencies=[Depends(check_api_key)])
async def extract(file: UploadFile = File(...)):
    """PDF -> 페이지별 텍스트."""
    with stage("PDF 읽기", "올바른 PDF 파일인지, 암호가 걸려 있지 않은지 확인하세요."):
        data = await file.read()
        if not data:
            raise StageError("PDF 읽기", "업로드된 파일이 비어 있습니다.", "파일을 다시 선택하세요.", 400)
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise StageError("PDF 읽기", "암호화된 PDF 입니다.", "암호를 해제한 PDF 를 사용하세요.", 400)
        pages = [{"page": i + 1, "text": (p.extract_text() or "").strip()}
                 for i, p in enumerate(reader.pages)]
    if not any(p["text"] for p in pages):
        raise StageError("텍스트 변환", "PDF 에서 추출된 텍스트가 없습니다.",
                         "스캔(이미지) PDF 일 수 있습니다. OCR 처리 후 텍스트가 포함된 PDF 를 사용하세요.", 422)
    return {"ok": True, "filename": file.filename, "page_count": len(pages), "pages": pages}


class Page(BaseModel):
    page: int
    text: str


class IngestRequest(BaseModel):
    source: str          # 파일명 (출처 표기에 사용)
    pages: List[Page]


@app.post("/ingest", dependencies=[Depends(check_api_key)])
def ingest(req: IngestRequest):
    """청크 분할 -> 임베딩 -> Supabase 저장. 진행 상황을 SSE 로 전달."""

    def worker(q):
        chunk_size = int(env("CHUNK_SIZE", "800"))
        overlap = int(env("CHUNK_OVERLAP", "120"))

        # 1) 청크 분할 (페이지 번호를 유지해 출처 표기에 사용)
        q.put({"type": "progress", "stage": "청크 분할", "percent": 5, "message": "텍스트를 청크로 분할하는 중"})
        with stage("청크 분할", "CHUNK_SIZE 가 CHUNK_OVERLAP 보다 큰지 확인하세요."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            chunks = []
            for p in req.pages:
                for piece in splitter.split_text(p.text):
                    if piece.strip():
                        chunks.append({"page": p.page, "content": piece})
        if not chunks:
            raise StageError("청크 분할", "분할 결과 청크가 0개입니다.", "PDF 텍스트가 비어 있는지 확인하세요.", 422)
        q.put({"type": "progress", "stage": "청크 분할", "percent": 10,
               "message": f"{len(chunks)}개 청크 생성 (크기 {chunk_size}, 겹침 {overlap})"})

        # 2) 같은 파일의 기존 청크 삭제 (재생성 시 중복 방지)
        with stage("기존 데이터 삭제", "Supabase URL/서비스 키와 documents 테이블 존재 여부를 확인하세요."):
            get_supabase().table("documents").delete().eq("source", req.source).execute()

        # 3) 배치 단위 임베딩 + 저장
        batch = 32
        embeddings = get_embeddings()
        for start in range(0, len(chunks), batch):
            part = chunks[start:start + batch]
            with stage("임베딩", "OPENAI_API_KEY 값과 OpenAI 사용 한도를 확인하세요."):
                vectors = embeddings.embed_documents([c["content"] for c in part])
            rows = [{"source": req.source, "page": c["page"], "chunk_index": start + i,
                     "content": c["content"], "embedding": v}
                    for i, (c, v) in enumerate(zip(part, vectors))]
            with stage("Supabase 저장",
                       "임베딩 차원(기본 1536)이 supabase.sql 의 vector(1536) 과 같은지 확인하세요."):
                get_supabase().table("documents").insert(rows).execute()
            done = start + len(part)
            q.put({"type": "progress", "stage": "임베딩/저장",
                   "percent": 10 + int(90 * done / len(chunks)),
                   "message": f"{done}/{len(chunks)} 청크 저장 완료"})

        q.put({"type": "done", "chunks": len(chunks), "source": req.source})

    return sse_stream(worker)


class ChatTurn(BaseModel):
    role: str            # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: List[ChatTurn] = []
    source: Optional[str] = None   # 지정하면 해당 파일에서만 검색


@app.post("/query", dependencies=[Depends(check_api_key)])
def query(req: QueryRequest):
    """질의 확장 -> 검색 -> 답변 스트리밍."""

    def worker(q):
        question = req.question.strip()
        if not question:
            raise StageError("입력 검증", "질문이 비어 있습니다.", "질문을 입력하세요.", 400)

        # 1) 질의 확장 체인: 프롬프트 | LLM | 문자열 파서
        q.put({"type": "status", "stage": "질의 확장", "message": "대화 맥락으로 질문을 명확하게 변환하는 중"})
        chat_history = [HumanMessage(t.content) if t.role == "user" else AIMessage(t.content)
                        for t in req.history]
        augment_prompt = ChatPromptTemplate.from_messages([
            ("system", AUGMENT_SYSTEM),
            MessagesPlaceholder("chat_history"),   # 과거 대화 기록이 들어갈 동적 빈칸
            ("human", "{question}"),
        ])
        query_augmentation_chain = augment_prompt | get_llm(streaming=False) | StrOutputParser()
        with stage("질의 확장", "GOOGLE_API_KEY 와 GEMINI_MODEL 이름이 올바른지 확인하세요."):
            augmented_query = query_augmentation_chain.invoke(
                {"question": question, "chat_history": chat_history}).strip() or question
        q.put({"type": "augmented", "query": augmented_query})

        # 2) 증강된 질의로 벡터 검색 (Supabase RPC: match_documents)
        q.put({"type": "status", "stage": "문서 검색", "message": "벡터 DB 에서 관련 청크를 검색하는 중"})
        with stage("질의 임베딩", "OPENAI_API_KEY 값과 사용 한도를 확인하세요."):
            vector = get_embeddings().embed_query(augmented_query)
        with stage("벡터 검색", "supabase.sql 의 match_documents 함수가 생성되었는지 확인하세요."):
            res = get_supabase().rpc("match_documents", {
                "query_embedding": vector,
                "match_count": int(env("TOP_K", "5")),
                "filter_source": req.source,
            }).execute()
        docs = res.data or []
        sources = [{"no": i + 1, "source": d["source"], "page": d["page"],
                    "similarity": round(d["similarity"], 4), "content": d["content"]}
                   for i, d in enumerate(docs)]
        q.put({"type": "sources", "sources": sources})
        if not docs:
            raise StageError("문서 검색", "검색된 문서가 없습니다.",
                             "먼저 PDF 로 '벡터 DB 생성'을 실행했는지 확인하세요.", 404)

        # 3) 검색된 청크만 근거로 답변 생성 (StreamHandler 로 토큰 단위 스트리밍)
        q.put({"type": "status", "stage": "답변 생성", "message": "Gemini 가 답변을 생성하는 중"})
        context = "\n\n".join(f"[{s['no']}] ({s['source']} p.{s['page']})\n{s['content']}" for s in sources)
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])
        answer_chain = answer_prompt | get_llm(streaming=True) | StrOutputParser()
        with stage("답변 생성", "Gemini 사용 한도/모델명을 확인하세요."):
            answer_chain.invoke(
                {"context": context, "question": augmented_query, "chat_history": chat_history},
                config={"callbacks": [StreamHandler(q)]},
            )
        q.put({"type": "done"})

    return sse_stream(worker)
