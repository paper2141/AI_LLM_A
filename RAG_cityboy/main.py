import os
import io
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2

# LangChain 관련 모듈 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from dotenv import load_dotenv

# 환경변수 로드 (로컬 개발용. Cloud Run에서는 시스템 환경변수 사용)
load_dotenv()

app = FastAPI(title="RAG AI Server")

# 클라이언트(App Script/HTML)에서 직접 접근할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 초기화 및 환경변수 검증
# ---------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Service Role Key 권장
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY, GOOGLE_API_KEY]):
    print("경고: 필요한 환경 변수(SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)가 모두 설정되지 않았습니다.")

# 클라이언트 인스턴스 생성 (요청 시 초기화해도 되지만 전역으로 유지)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # 오픈 AI 임베딩 모델 사용하기
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"초기화 에러: {str(e)}")

# API 엔드포인트용 데이터 모델
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] # [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

class IndexRequest(BaseModel):
    text: str
    filename: str

# ---------------------------------------------------------
# API 1: 헬스 체크
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    """서버 연동 상태를 확인하는 엔드포인트"""
    return {"status": "ok", "message": "서버가 정상적으로 실행 중입니다."}

# ---------------------------------------------------------
# API 2: PDF 파일 텍스트 추출
# ---------------------------------------------------------
@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """입력한 PDF 파일을 텍스트로 변환하는 기능"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
        return {"filename": file.filename, "text": text}
    
    except Exception as e:
        # 에러 메시지를 세밀하게 표현
        raise HTTPException(status_code=500, detail=f"PDF 텍스트 추출 중 오류가 발생했습니다: {str(e)}")

# ---------------------------------------------------------
# API 3: 텍스트 청크 분할 및 벡터 DB(Supabase) 저장
# ---------------------------------------------------------
@app.post("/create-rag")
async def create_rag(req: IndexRequest):
    """추출된 텍스트를 청크 단위로 쪼개고 Supabase에 RAG 인덱싱 수행"""
    try:
        # 1. 텍스트를 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(req.text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="추출된 텍스트가 없어 청크를 생성할 수 없습니다.")
            
        metadatas = [{"source": req.filename, "chunk_index": i} for i in range(len(chunks))]
        
        # 2. 벡트 DB 개발: Supabase에 저장
        SupabaseVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            client=supabase,
            metadatas=metadatas,
            table_name="documents",
            query_name="match_documents" # 생성한 SQL 함수 이름
        )
        
        return {"status": "success", "message": f"총 {len(chunks)}개의 청크가 Vector DB에 성공적으로 저장되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB 생성 중 오류가 발생했습니다: {str(e)}")

# ---------------------------------------------------------
# API 4: 스트리밍 채팅 (질의 확장 + RAG + 스트리밍 핸들러)
# ---------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    1. 질의 확장 구현 (Query Augmentation)
    2. Supabase 리트리버로 문서 검색
    3. Gemini LLM을 통한 답변 생성 (실시간 스트리밍 반환)
    """
    try:
        # 1) 대화 기록(Chat History) 객체 변환 (LangChain 포맷)
        history = []
        for msg in req.chat_history:
            if msg.get('role') == 'user':
                history.append(HumanMessage(content=msg.get('content', '')))
            else:
                history.append(AIMessage(content=msg.get('content', '')))
                
        # LLM 초기화 (gemini-3.5-flash-lite 모델 사용)
        # LangChain의 최신 기능인 스트리밍 지원 모델 사용
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite", 
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        
        # ==========================================
        # STEP 1: Query Augmentation Chain 구성
        # ==========================================
        # 1. 시스템 메시지 및 프롬프트 설계
        augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"),
            # 2. 대화 기록 전달 공간 마련 (MessagesPlaceholder)
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # 3. Chain 파이프라인 연결 (Prompt -> LLM -> StrOutputParser)
        query_augmentation_chain = augmentation_prompt | llm | StrOutputParser()
        
        # 4. 체인 실행 (데이터 주입)
        if history:
            # 이전 대화가 있을 경우 질의 확장 실행
            augmented_query = await query_augmentation_chain.ainvoke({
                "chat_history": history,
                "question": req.query
            })
        else:
            # 첫 질문일 경우 질문 그대로 사용
            augmented_query = req.query
            
        # ==========================================
        # STEP 2: 벡터 DB 검색 (Supabase Retriever)
        # ==========================================
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        # 5. 증강된 질의로 벡터 DB 검색
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(augmented_query)
        
        # 검색된 컨텍스트 조립 및 메타데이터 추출 (출처 표기용)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = [{"content": doc.page_content, "source": doc.metadata.get('source', 'Unknown')} for doc in retrieved_docs]
        
        # ==========================================
        # STEP 3: 최종 답변 생성 (Streaming)
        # ==========================================
        # 시스템 메시지: 도시 정책 전문가
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 반드시 아래 제공된 '검색된 문서'를 바탕으로 답변을 작성해. 문서에 없는 내용은 지어내지 마."),
            ("human", "검색된 문서:\n{context}\n\n사용자 질문: {question}")
        ])
        
        rag_chain = rag_prompt | llm | StrOutputParser()
        
        # 스트리밍 응답 제너레이터 함수 (StreamHandler 역할을 대체하는 비동기 스트림)
        async def generate_stream():
            # 첫 번째 청크: 검색된 출처 데이터와 확장된 쿼리를 먼저 클라이언트에 전송
            init_data = {
                "type": "sources",
                "augmented_query": augmented_query,
                "sources": sources
            }
            yield f"data: {json.dumps(init_data, ensure_ascii=False)}\n\n"
            
            # 두 번째 청크부터는 LLM의 스트리밍 응답을 전송 (astream 사용)
            try:
                async for chunk in rag_chain.astream({
                    "context": context_text,
                    "question": augmented_query
                }):
                    chunk_data = {"type": "answer_chunk", "content": chunk}
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
                # 스트림 종료 시그널
                yield "data: [DONE]\n\n"
            except Exception as stream_e:
                error_data = {"type": "error", "content": f"스트리밍 중 에러 발생: {str(stream_e)}"}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                
        # Server-Sent Events(SSE) 포맷으로 스트리밍 반환
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류가 발생했습니다: {str(e)}")
