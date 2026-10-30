"""
서울시 도시 계획가 RAG 챗봇 서버 (Google Cloud Run / FastAPI)

RAG 구성
  - 문서 로더      : pypdf 로 PDF 읽기
  - 임베딩 모델    : Google 임베딩 (GoogleGenerativeAIEmbeddings)
  - 벡터 DB        : Chroma DB
  - 검색기         : 다중 질문 검색기 (MultiQueryRetriever)
  - 질의 확장      : RunnablePassthrough 를 이용한 체인 (대화 기록으로 질문을 명확하게 변환)
  - LLM            : Gemini (기본 gemini-3.5-flash-lite), StreamHandler 로 토큰 스트리밍

엔드포인트
  GET  /health   서버 상태 + RAG 설정 내용
  POST /extract  PDF -> 페이지별 텍스트
  POST /ingest   청크 분할 -> 임베딩 -> Chroma 저장 (진행 상황 SSE)
  POST /query    질의 확장 -> 다중 질문 검색 -> 답변 스트리밍 (SSE)

환경 변수 (코드에 키를 직접 쓰지 않습니다)
  GOOGLE_API_KEY   Google AI API 키 (LLM + 임베딩 공용)
선택
  GEMINI_MODEL     기본 gemini-3.5-flash-lite
  EMBEDDING_MODEL  기본 models/gemini-embedding-001
  CHROMA_DIR       Chroma 저장 폴더 (기본 /tmp/chroma)
  API_SECRET       설정하면 모든 요청에 X-API-Key 헤더가 일치해야 함
  TOP_K            검색 청크 수 (기본 4)
  CHUNK_SIZE / CHUNK_OVERLAP  청크 크기/겹침 (기본 800 / 120)

주의: Cloud Run 의 파일 시스템은 임시 공간이라 인스턴스가 재시작되면 Chroma 데이터가 사라집니다.
      --max-instances=1 로 배포하고, 재시작 후에는 '벡터 DB 생성'을 다시 실행하세요.
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

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-server")

app = FastAPI(title="Seoul Urban Planner RAG Server")

# --------------------------------------------------------------------------
# 프롬프트
# --------------------------------------------------------------------------

# 질의 확장용 시스템 메시지: 대화 맥락을 이용해 대명사를 명확한 명사로 바꾼 한 문장 질문을 만든다.
AUGMENT_SYSTEM = (
    "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
    "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. "
    "질문에 대한 답변은 하지 말고, 변환된 질문 한 문장만 출력하라."
)

# 다중 질문 검색기용 프롬프트: 한 질문을 서로 다른 관점의 질문 3개로 바꿔 검색 누락을 줄인다.
MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "당신은 벡터 DB 검색을 돕는 도우미입니다. 아래 질문을 서로 다른 관점의 한국어 질문 3개로 바꿔 쓰세요. "
    "한 줄에 하나씩, 번호나 설명 없이 질문만 출력하세요.\n\n원래 질문: {question}"
)

# 답변 생성용 시스템 메시지 (시험에서 지정한 역할 정의 + 문서 기반 답변 규칙)
ANSWER_SYSTEM = (
    "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요\n"
    "아래 [문서] 에 있는 내용을 근거로 답변하고, 문서에 없는 내용은 추측하지 말고 "
    "'제공된 문서에서 해당 내용을 찾을 수 없습니다'라고 답하세요.\n"
    "근거가 된 문서는 답변 끝에 [출처 번호] 형식으로 표기하세요.\n\n[문서]\n{context}"
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
# 모델 / 벡터 DB (최초 사용 시 생성 -> 키가 없어도 /health 는 응답 가능)
# --------------------------------------------------------------------------

LLM_MODEL = lambda: os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")  # noqa: E731
EMBED_MODEL = lambda: os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")  # noqa: E731


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL(), google_api_key=env("GOOGLE_API_KEY"))


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Chroma 컬렉션 (디스크에 저장)."""
    return Chroma(
        collection_name="seoul_urban_policy",
        embedding_function=get_embeddings(),
        persist_directory=env("CHROMA_DIR", "/tmp/chroma"),
    )


def get_llm(streaming: bool) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL(),
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
    """연동 상태 + RAG 설정 내용. (키 값은 노출하지 않음)"""
    missing = [] if os.environ.get("GOOGLE_API_KEY") else ["GOOGLE_API_KEY"]
    result = {
        "ok": not missing,
        "missing_env": missing,
        # 클라이언트의 'RAG 설정 내용' 표시에 사용
        "rag": {
            "vector_db": "Chroma DB",
            "embedding_model": EMBED_MODEL(),
            "llm_model": LLM_MODEL(),
            "retriever": "다중 질문 검색기 (MultiQueryRetriever)",
            "query_expansion": "RunnablePassthrough 기반 질의 확장 (대화 기록으로 질문 명료화)",
            "top_k": int(os.environ.get("TOP_K", "4")),
        },
        "chroma": None,
        "chunk_count": None,
    }
    if not missing:
        try:
            result["chunk_count"] = get_vectorstore()._collection.count()
            result["chroma"] = "connected"
        except Exception as e:  # noqa: BLE001
            result["ok"] = False
            result["chroma"] = f"{type(e).__name__}: {e}"
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
    """청크 분할 -> 임베딩 -> Chroma 저장. 진행 상황을 SSE 로 전달."""

    def worker(q):
        chunk_size = int(env("CHUNK_SIZE", "800"))
        overlap = int(env("CHUNK_OVERLAP", "120"))

        # 1) 청크 분할 (페이지 번호를 유지해 출처 표기에 사용)
        q.put({"type": "progress", "stage": "청크 분할", "percent": 5, "message": "텍스트를 청크로 분할하는 중"})
        with stage("청크 분할", "CHUNK_SIZE 가 CHUNK_OVERLAP 보다 큰지 확인하세요."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            docs = []
            for p in req.pages:
                for piece in splitter.split_text(p.text):
                    if piece.strip():
                        docs.append(Document(page_content=piece,
                                             metadata={"source": req.source, "page": p.page}))
        if not docs:
            raise StageError("청크 분할", "분할 결과 청크가 0개입니다.", "PDF 텍스트가 비어 있는지 확인하세요.", 422)
        q.put({"type": "progress", "stage": "청크 분할", "percent": 10,
               "message": f"{len(docs)}개 청크 생성 (크기 {chunk_size}, 겹침 {overlap})"})

        # 2) 같은 파일의 기존 청크 삭제 (재생성 시 중복 방지)
        with stage("기존 데이터 삭제", "CHROMA_DIR 경로에 쓰기 권한이 있는지 확인하세요."):
            get_vectorstore()._collection.delete(where={"source": req.source})

        # 3) 배치 단위 임베딩 + Chroma 저장
        batch = 20
        for start in range(0, len(docs), batch):
            part = docs[start:start + batch]
            with stage("임베딩/저장",
                       "GOOGLE_API_KEY 값, 임베딩 모델명(EMBEDDING_MODEL), 구글 API 사용 한도를 확인하세요."):
                get_vectorstore().add_documents(part)
            done = start + len(part)
            q.put({"type": "progress", "stage": "임베딩/저장",
                   "percent": 10 + int(90 * done / len(docs)),
                   "message": f"{done}/{len(docs)} 청크 저장 완료"})

        q.put({"type": "done", "chunks": len(docs), "source": req.source})

    return sse_stream(worker)


class ChatTurn(BaseModel):
    role: str            # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: List[ChatTurn] = []


@app.post("/query", dependencies=[Depends(check_api_key)])
def query(req: QueryRequest):
    """질의 확장 -> 다중 질문 검색 -> 답변 스트리밍."""

    def worker(q):
        question = req.question.strip()
        if not question:
            raise StageError("입력 검증", "질문이 비어 있습니다.", "질문을 입력하세요.", 400)

        # 1) 질의 확장: RunnablePassthrough.assign 으로 원래 입력을 유지한 채 augmented_query 를 추가
        q.put({"type": "status", "stage": "질의 확장", "message": "대화 맥락으로 질문을 명확하게 변환하는 중"})
        chat_history = [HumanMessage(t.content) if t.role == "user" else AIMessage(t.content)
                        for t in req.history]
        augment_prompt = ChatPromptTemplate.from_messages([
            ("system", AUGMENT_SYSTEM),
            MessagesPlaceholder("chat_history"),   # 과거 대화 기록이 들어갈 동적 빈칸
            ("human", "{question}"),
        ])
        llm_plain = get_llm(streaming=False)
        expansion_chain = RunnablePassthrough.assign(
            augmented_query=augment_prompt | llm_plain | StrOutputParser()
        )
        with stage("질의 확장", "GOOGLE_API_KEY 와 GEMINI_MODEL 이름이 올바른지 확인하세요."):
            expanded = expansion_chain.invoke({"question": question, "chat_history": chat_history})
        augmented_query = expanded["augmented_query"].strip() or question
        q.put({"type": "augmented", "query": augmented_query})

        # 2) 다중 질문 검색: 증강된 질문을 여러 관점의 질문으로 바꿔 각각 검색한 뒤 중복 제거
        q.put({"type": "status", "stage": "문서 검색", "message": "다중 질문 검색기로 Chroma 에서 검색하는 중"})
        with stage("Chroma 연결", "먼저 PDF 로 '벡터 DB 생성'을 실행했는지 확인하세요."):
            vs = get_vectorstore()
            if vs._collection.count() == 0:
                raise StageError("문서 검색", "Chroma 에 저장된 청크가 없습니다.",
                                 "'벡터 DB 생성'을 먼저 실행하세요. (Cloud Run 이 재시작되면 데이터가 초기화됩니다)", 404)
            base_retriever = vs.as_retriever(search_kwargs={"k": int(env("TOP_K", "4"))})
            multi_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever, llm=llm_plain, prompt=MULTI_QUERY_PROMPT)
        with stage("다중 질문 생성", "Gemini 사용 한도/모델명을 확인하세요."):
            generated = multi_retriever.llm_chain.invoke({"question": augmented_query})
            queries = [augmented_query] + [s.strip() for s in generated if s.strip()]
        q.put({"type": "queries", "queries": queries})
        with stage("벡터 검색", "임베딩 모델명과 GOOGLE_API_KEY 를 확인하세요."):
            found = [d for qs in queries for d in base_retriever.invoke(qs)]
            docs = multi_retriever.unique_union(found)
        sources = [{"no": i + 1, "source": d.metadata.get("source"), "page": d.metadata.get("page"),
                    "content": d.page_content} for i, d in enumerate(docs)]
        q.put({"type": "sources", "sources": sources})
        if not docs:
            raise StageError("문서 검색", "검색된 문서가 없습니다.", "다른 표현으로 질문해 보세요.", 404)

        # 3) 검색된 청크를 근거로 답변 생성 (StreamHandler 로 토큰 단위 스트리밍)
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
