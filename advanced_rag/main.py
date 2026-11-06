"""
================================================================================
Advanced RAG Backend Server (main.py)
- 아키텍처: FAISS Index + Multi-Query + Hybrid Search (Dense + BM25) + RAG Fusion (RRF)
- 프레임워크: FastAPI (Google Cloud Run 배포 및 Google Apps Script 연동)
- 공용 가상환경: c:\\Users\\piayc\\ai project\\ai-class-env\\Scripts\\python.exe

[알고리즘 시간/공간 복잡도 요약]
1. 문서 청킹 및 전처리 (Chunking & Tokenization):
   - 시간 복잡도: O(L) (L: 전체 문서 문자 길이)
   - 공간 복잡도: O(N * C) (N: 청크 수, C: 평균 청크 길이)
2. FAISS 인덱스 구축 및 검색 (Dense Vector Retrieval):
   - IndexFlatL2 (전수 L2 거리 계산):
     * 구축 시간: O(N * d), 검색 시간: O(N * d), 공간 복잡도: O(N * d) (d: 임베딩 차원)
   - IndexHNSWFlat (계층적 탐색 가능 작은 세상 그래프):
     * 구축 시간: O(N * d * log N), 검색 시간: O(d * log N), 공간 복잡도: O(N * (d + M)) (M: 그래프 연결 수)
3. BM25 키워드 검색 (Sparse Retrieval - BM25Okapi):
   - 구축 시간: O(N * T_avg) (역색인 생성, T_avg: 청크당 평균 토큰 수)
   - 검색 시간: O(Q_t * N_postings) (최악 O(Q_t * N), Q_t: 질의 토큰 수)
   - 공간 복잡도: O(V + N * T_avg) (V: 고유 어휘 수)
4. 다중 쿼리 확장 (Multi-Query Expansion):
   - 시간 복잡도: O(1) LLM API 호출 (생성 쿼리 수 M_q개, 토큰 생성 비용 O(T_gen))
5. 하이브리드 검색 + RRF 순위 융합 (RAG Fusion & Reciprocal Rank Fusion):
   - 하이브리드 검색 시간: O(M_q * (T_dense + T_bm25))
   - RRF 융합 및 재정렬 시간: O(M_q * K * log(M_q * K)) (K: 각 검색기 Top-K 문서 수)
   - RRF 공식: RRF_score(d) = sum_{r in Retrievers} 1 / (k_rrf + rank_r(d)) (기본 k_rrf = 60)
================================================================================
"""

import base64
import hashlib
import io
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Google Gemini SDK (선택적 로드 및 환경변수 기반 활성화)
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ==============================================================================
# 1. 데이터 모델 및 스키마 정의
# ==============================================================================
@dataclass
class DocumentChunk:
    """개별 문서 청크 및 출처 메타데이터 구조체"""
    doc_id: str
    chunk_id: int
    source_file: str
    page_number: int
    text: str
    char_count: int
    token_count: int


class IngestBase64Request(BaseModel):
    """GAS(UrlFetchApp)에서 PDF 파일을 Base64로 전송할 때 사용하는 요청 스키마"""
    filename: str = Field(..., description="업로드할 PDF 파일명")
    file_base64: str = Field(..., description="Base64 인코딩된 PDF 바이너리 데이터")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="청크 최대 글자 수")
    chunk_overlap: int = Field(default=80, ge=0, le=500, description="청크 간 중첩 글자 수")
    index_type: str = Field(default="IndexHNSWFlat", description="FAISS 인덱스 종류 (IndexFlatL2 또는 IndexHNSWFlat)")
    stream: bool = Field(default=True, description="StreamHandler 호환 스트리밍 이벤트 포함 여부")


class QueryRequest(BaseModel):
    """질의 및 RAG Fusion 파이프라인 실행 요청 스키마"""
    query: str = Field(..., min_length=1, description="사용자 원본 질의")
    top_k: int = Field(default=5, ge=1, le=20, description="최종 반환할 상위 문서 개수")
    num_queries: int = Field(default=3, ge=1, le=6, description="Multi-Query 파생 쿼리 생성 개수")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF 수식의 상수 k 값 (기본 60)")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 Sparse 검색 가중치")
    stream: bool = Field(default=True, description="StreamHandler용 단계별 스트리밍 로그 반환 여부")


# ==============================================================================
# 2. 실시간 스트리밍 핸들러 (StreamHandler)
# ==============================================================================
class StreamHandler:
    """
    서버 파이프라인의 실시간 진행 현황(단계, 진행률, 복잡도, 상태 메시지)을
    기록하고 NDJSON(Newline-Delimited JSON) 스트림 또는 이벤트 배열로 송출하는 핸들러.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.start_time: float = time.perf_counter()

    def emit(
        self,
        stage: str,
        progress: int,
        message: str,
        complexity: str = "",
        detail: Optional[Dict[str, Any]] = None,
        status: str = "PROCESSING",
    ) -> Dict[str, Any]:
        """단일 스트리밍 이벤트를 생성하고 버퍼에 기록합니다."""
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        event = {
            "type": "stream_event",
            "stage": stage,
            "progress": max(0, min(100, progress)),
            "message": message,
            "complexity": complexity,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "detail": detail or {},
        }
        self.events.append(event)
        return event

    def format_ndjson_line(self, event: Dict[str, Any]) -> str:
        """GAS StreamHandler 및 HTTP 스트리밍 클라이언트가 파싱할 수 있는 NDJSON 라인 포맷"""
        return json.dumps(event, ensure_ascii=False) + "\n"


# ==============================================================================
# 3. 핵심 RAG 엔진 (FAISS + BM25 + Multi-Query + RRF Fusion)
# ==============================================================================
class AdvancedRAGEngine:
    """
    FAISS(Dense) + BM25(Sparse) 하이브리드 인덱스와
    Multi-Query 확장 및 RRF(Reciprocal Rank Fusion) 재정렬을 수행하는 엔진 클래스
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        self.embedding_dim: int = embedding_dim
        self.index_type: str = "IndexHNSWFlat"
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.loaded_files: List[Dict[str, Any]] = []
        self.last_updated: Optional[str] = None

        # Gemini 클라이언트 초기화 (환경변수 GEMINI_API_KEY 확인)
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.genai_client = None
        if HAS_GENAI and self.api_key:
            try:
                self.genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                self.genai_client = None

    # --------------------------------------------------------------------------
    # [전처리/토큰화] 한국어 및 영문 혼합 형태소/단어 토크나이저
    # 시간 복잡도: O(L) (L: 입력 문자열 길이)
    # --------------------------------------------------------------------------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        BM25 검색 및 로컬 임베딩을 위한 단어+바이그램 혼합 토크나이저.
        한국어 조사 분리를 보완하기 위해 어절 토큰과 2-gram 음절 토큰을 함께 추출합니다.
        """
        cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) > 1]
        bigrams: List[str] = []
        for word in words:
            if re.search(r"[가-힣]", word) and len(word) >= 3:
                for i in range(len(word) - 1):
                    bigrams.append(word[i : i + 2])
        return words + bigrams

    # --------------------------------------------------------------------------
    # [임베딩 생성] Gemini Embedding 또는 결정론적 고차원 로컬 임베딩 폴백
    # 시간 복잡도: O(N * d) (N: 텍스트 개수, d: 임베딩 차원)
    # --------------------------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 고정 차원(self.embedding_dim) L2 정규화 실수 벡터(float32)로 변환합니다.
        Gemini API 키가 설정된 경우 text-embedding-004 모델을 호출하며,
        미설정 또는 오프라인 환경에서는 Feature Hashing 기반 밀집 투영 벡터를 생성합니다.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self.genai_client is not None:
            try:
                res = self.genai_client.models.embed_content(
                    model="text-embedding-004",
                    contents=texts,
                )
                vectors = np.array([emb.values for emb in res.embeddings], dtype=np.float32)
                if vectors.shape[1] != self.embedding_dim:
                    self.embedding_dim = vectors.shape[1]
                faiss.normalize_L2(vectors)
                return vectors
            except Exception:
                # API 할당량 초과 등 발생 시 로컬 결정론적 임베딩으로 안전하게 폴백
                pass

        # 결정론적 Feature Hashing (Signed Random Projection) 벡터 생성
        matrix = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = self.tokenize(text)
            if not tokens:
                tokens = [text.strip() or "empty"]
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx1 = int.from_bytes(digest[0:4], "little") % self.embedding_dim
                idx2 = int.from_bytes(digest[4:8], "little") % self.embedding_dim
                sign1 = 1.0 if (digest[8] & 1) == 0 else -1.0
                sign2 = 1.0 if (digest[9] & 1) == 0 else -1.0
                matrix[row_idx, idx1] += sign1
                matrix[row_idx, idx2] += sign2 * 0.5

        # L2 정규화 (유클리드 거리가 코사인 유사도와 단조 관계를 갖도록 정규화)
        faiss.normalize_L2(matrix)
        return matrix

    # --------------------------------------------------------------------------
    # [PDF 파싱 및 문서 청킹]
    # 시간 복잡도: O(P * L_p) (P: 페이지 수, L_p: 페이지당 문자 수)
    # --------------------------------------------------------------------------
    def extract_and_chunk_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> Tuple[List[DocumentChunk], int]:
        """PDF 바이너리에서 페이지별 텍스트를 추출하고 슬라이딩 윈도우 방식으로 청킹합니다."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as exc:
            raise ValueError(f"PDF 파일을 읽을 수 없습니다 ({filename}): {str(exc)}") from exc

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"PDF 파일({filename})에 페이지가 존재하지 않습니다.")

        extracted_chunks: List[DocumentChunk] = []
        step = max(50, chunk_size - chunk_overlap)
        chunk_counter = len(self.chunks)

        for page_idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            normalized = re.sub(r"\s+", " ", raw_text).strip()
            if not normalized:
                continue

            for start in range(0, len(normalized), step):
                segment = normalized[start : start + chunk_size].strip()
                if len(segment) < 20 and extracted_chunks:
                    continue
                tokens = self.tokenize(segment)
                chunk_counter += 1
                extracted_chunks.append(
                    DocumentChunk(
                        doc_id=f"{filename}_p{page_idx}_c{chunk_counter}",
                        chunk_id=chunk_counter,
                        source_file=filename,
                        page_number=page_idx,
                        text=segment,
                        char_count=len(segment),
                        token_count=len(tokens),
                    )
                )

        if not extracted_chunks:
            raise ValueError(
                f"PDF 파일({filename})에서 추출 가능한 텍스트가 없습니다. 스캔본(이미지) PDF인지 확인해 주세요."
            )

        return extracted_chunks, total_pages

    # --------------------------------------------------------------------------
    # [1단계: FAISS 인덱스 및 BM25 인덱스 구축]
    # - IndexFlatL2 복잡도: 구축 O(N * d), 검색 O(N * d)
    # - IndexHNSWFlat 복잡도: 구축 O(N * d * log N), 검색 O(d * log N)
    # --------------------------------------------------------------------------
    def build_indices(
        self,
        new_chunks: List[DocumentChunk],
        filename: str,
        total_pages: int,
        index_type: str,
        stream_handler: StreamHandler,
    ) -> Dict[str, Any]:
        """추출된 문서 청크를 기반으로 FAISS Dense Index와 BM25 Sparse Index를 구축합니다."""
        self.index_type = index_type if index_type in ("IndexFlatL2", "IndexHNSWFlat") else "IndexHNSWFlat"

        # 기존 청크를 초기화하고 새 PDF 기준으로 클린 빌드 (실습 명확성 보장)
        self.chunks = new_chunks

        stream_handler.emit(
            stage="EMBEDDING",
            progress=45,
            message=f"총 {len(self.chunks)}개 청크에 대한 고차원 벡터 임베딩 생성 중...",
            complexity="O(N * d) — N: 청크 수, d: 임베딩 차원",
        )
        texts = [c.text for c in self.chunks]
        vectors = self.embed_texts(texts)
        num_vectors, dim = vectors.shape

        stream_handler.emit(
            stage="FAISS_INDEXING",
            progress=70,
            message=f"FAISS 인덱스({self.index_type}, dim={dim}) 구축 및 벡터 적재 중...",
            complexity=(
                "O(N * d * log N) [IndexHNSWFlat]"
                if self.index_type == "IndexHNSWFlat"
                else "O(N * d) [IndexFlatL2]"
            ),
        )

        # FAISS 인덱스 생성 및 적재
        if self.index_type == "IndexHNSWFlat":
            # HNSW 그래프 인덱스: M=32 연결, efConstruction=64, efSearch=32
            hnsw_index = faiss.IndexHNSWFlat(dim, 32)
            hnsw_index.hnsw.efConstruction = 64
            hnsw_index.hnsw.efSearch = 32
            hnsw_index.add(vectors)
            self.faiss_index = hnsw_index
        else:
            flat_index = faiss.IndexFlatL2(dim)
            flat_index.add(vectors)
            self.faiss_index = flat_index

        stream_handler.emit(
            stage="BM25_INDEXING",
            progress=88,
            message="BM25Okapi 키워드 역색인(Sparse Index) 구축 중...",
            complexity="O(N * T_avg) — T_avg: 청크당 평균 토큰 수",
        )
        self.tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        file_meta = {
            "filename": filename,
            "pages": total_pages,
            "chunks": len(new_chunks),
            "index_type": self.index_type,
            "embedding_dim": dim,
            "indexed_at": self.last_updated,
        }
        self.loaded_files = [file_meta]

        stream_handler.emit(
            stage="COMPLETED",
            progress=100,
            message=f"벡터 DB 구축 완료! ({filename}: {total_pages}페이지, {len(new_chunks)}청크, {self.index_type})",
            complexity="FAISS Search: O(d log N) / BM25 Search: O(Q_t * N)",
            detail=file_meta,
            status="READY",
        )
        return file_meta

    # --------------------------------------------------------------------------
    # [2단계: 다중 쿼리 확장 계층 (Multi-Query Layer)]
    # 단일 쿼리를 다양한 관점의 파생 쿼리로 확장하여 재현율(Recall) 향상
    # 시간 복잡도: O(1) LLM 호출 (생성 토큰 수에 비례)
    # --------------------------------------------------------------------------
    def generate_multi_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """
        사용자의 단일 질의를 서로 다른 관점(의미적 패러프레이징, 핵심 키워드 중심,
        구체적 조건/배경 중심)의 파생 쿼리들로 확장합니다.
        """
        queries: List[str] = [original_query.strip()]
        if num_queries <= 1:
            return queries

        if self.genai_client is not None:
            try:
                prompt = (
                    "당신은 RAG 파이프라인의 Multi-Query 확장 전문가입니다.\n"
                    f"사용자의 원본 질의에 대해 검색 재현율(Recall)을 높일 수 있도록 서로 다른 관점의 파생 검색 쿼리를 {num_queries}개 생성하세요.\n"
                    "- 관점 1: 핵심 전문 용어 및 키워드 중심 쿼리\n"
                    "- 관점 2: 개념적 정의 및 배경 설명 중심 쿼리\n"
                    "- 관점 3: 구체적인 수치/절차/사례 확인 중심 쿼리\n"
                    "반드시 줄바꿈으로만 구분하여 쿼리 문장만 출력하세요 (번호나 불릿 기호 제외).\n\n"
                    f"원본 질의: {original_query}"
                )
                response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                if response and response.text:
                    lines = [
                        re.sub(r"^[\d\-\*\.\)\s]+", "", line).strip()
                        for line in response.text.strip().splitlines()
                        if line.strip()
                    ]
                    for line in lines:
                        if line and line not in queries:
                            queries.append(line)
                        if len(queries) >= num_queries + 1:
                            break
                    if len(queries) > 1:
                        return queries[: num_queries + 1]
            except Exception:
                pass

        # 규칙 기반 다중 관점 쿼리 확장 (LLM API 키 미설정 시에도 Multi-Query 동작 보장)
        core_tokens = [w for w in re.sub(r"[^\w\s가-힣]", " ", original_query).split() if len(w) >= 2]
        keyword_focus = " ".join(core_tokens) if core_tokens else original_query

        perspective_templates = [
            f"{keyword_focus} 핵심 개념 정의 및 주요 특징",
            f"{original_query} 관련 상세 원리 및 구성 요소",
            f"{keyword_focus} 실제 적용 방법 및 핵심 요약",
            f"{original_query} 장단점 비교 및 주요 메커니즘",
        ]
        for candidate in perspective_templates:
            if len(queries) >= num_queries + 1:
                break
            if candidate not in queries:
                queries.append(candidate)

        return queries[: num_queries + 1]

    # --------------------------------------------------------------------------
    # [3단계: 하이브리드 검색 계층 (FAISS Dense + BM25 Sparse)]
    # --------------------------------------------------------------------------
    def search_single_query(
        self, query_text: str, candidate_k: int = 10
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        단일 쿼리에 대해:
        1) FAISS 인덱스에서 L2 거리 기반 상위 candidate_k개 문서를 검색 (Dense)
        2) BM25 인덱스에서 키워드 매칭 점수 기반 상위 candidate_k개 문서를 검색 (Sparse)
        반환값: (dense_results[(chunk_idx, distance)], sparse_results[(chunk_idx, bm25_score)])
        """
        if self.faiss_index is None or self.bm25_index is None or not self.chunks:
            raise RuntimeError("벡터 DB가 아직 구축되지 않았습니다. 먼저 PDF 문서를 업로드하여 벡터 DB를 생성해 주세요.")

        total_chunks = len(self.chunks)
        k = min(candidate_k, total_chunks)

        # 1. FAISS Dense Vector 검색
        q_vec = self.embed_texts([query_text])
        distances, indices = self.faiss_index.search(q_vec, k)
        dense_results: List[Tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= int(idx) < total_chunks:
                dense_results.append((int(idx), float(dist)))

        # 2. BM25 Sparse Keyword 검색
        q_tokens = self.tokenize(query_text)
        bm25_scores = self.bm25_index.get_scores(q_tokens)
        ranked_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        sparse_results: List[Tuple[int, float]] = [
            (int(idx), float(bm25_scores[idx])) for idx in ranked_bm25_indices
        ]

        return dense_results, sparse_results

    # --------------------------------------------------------------------------
    # [4단계: RRF 순위 융합 계층 (RAG Fusion & Reciprocal Rank Fusion)]
    # 수식: RRF_score(d) = sum_{q in Queries} (w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d)))
    # 시간 복잡도: O(M_q * K * log(M_q * K))
    # --------------------------------------------------------------------------
    def rag_fusion_rrf(
        self,
        expanded_queries: List[str],
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        확장된 다중 쿼리 각각에 대해 FAISS(Dense)와 BM25(Sparse) 검색을 수행한 뒤,
        Reciprocal Rank Fusion(상호 순위 융합) 공식으로 점수를 합산하고 재정렬합니다.
        """
        candidate_k = max(top_k * 2, 10)
        rrf_scores: Dict[int, float] = {}
        doc_rank_trace: Dict[int, Dict[str, Any]] = {}
        per_query_diagnostics: List[Dict[str, Any]] = []

        # 가중치 정규화
        total_w = dense_weight + sparse_weight
        w_d = (dense_weight / total_w) if total_w > 0 else 0.5
        w_s = (sparse_weight / total_w) if total_w > 0 else 0.5

        for q_idx, q_text in enumerate(expanded_queries):
            dense_res, sparse_res = self.search_single_query(q_text, candidate_k=candidate_k)

            per_query_diagnostics.append(
                {
                    "query_index": q_idx,
                    "query_text": q_text,
                    "is_original": q_idx == 0,
                    "dense_top_doc": self.chunks[dense_res[0][0]].doc_id if dense_res else None,
                    "sparse_top_doc": self.chunks[sparse_res[0][0]].doc_id if sparse_res else None,
                }
            )

            # Dense 검색 순위 반영 (1-based rank)
            for rank_pos, (chunk_idx, l2_dist) in enumerate(dense_res, start=1):
                contribution = w_d * (1.0 / (rrf_k + rank_pos))
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + contribution

                trace = doc_rank_trace.setdefault(
                    chunk_idx,
                    {
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "min_l2_distance": None,
                        "max_bm25_score": None,
                        "matched_queries": set(),
                        "dense_hits": 0,
                        "sparse_hits": 0,
                    },
                )
                if trace["best_dense_rank"] is None or rank_pos < trace["best_dense_rank"]:
                    trace["best_dense_rank"] = rank_pos
                if trace["min_l2_distance"] is None or l2_dist < trace["min_l2_distance"]:
                    trace["min_l2_distance"] = round(l2_dist, 4)
                trace["dense_hits"] += 1
                trace["matched_queries"].add(q_idx)

            # Sparse(BM25) 검색 순위 반영 (1-based rank)
            for rank_pos, (chunk_idx, bm25_val) in enumerate(sparse_res, start=1):
                contribution = w_s * (1.0 / (rrf_k + rank_pos))
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + contribution

                trace = doc_rank_trace.setdefault(
                    chunk_idx,
                    {
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "min_l2_distance": None,
                        "max_bm25_score": None,
                        "matched_queries": set(),
                        "dense_hits": 0,
                        "sparse_hits": 0,
                    },
                )
                if trace["best_sparse_rank"] is None or rank_pos < trace["best_sparse_rank"]:
                    trace["best_sparse_rank"] = rank_pos
                if trace["max_bm25_score"] is None or bm25_val > trace["max_bm25_score"]:
                    trace["max_bm25_score"] = round(bm25_val, 4)
                trace["sparse_hits"] += 1
                trace["matched_queries"].add(q_idx)

        # RRF 점수 내림차순으로 상위 문서 재정렬 (Re-ranking)
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        reranked_results: List[Dict[str, Any]] = []
        for final_rank, chunk_idx in enumerate(sorted_indices, start=1):
            chunk = self.chunks[chunk_idx]
            trace = doc_rank_trace[chunk_idx]
            reranked_results.append(
                {
                    "final_rank": final_rank,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "snippet": chunk.text,
                    "char_count": chunk.char_count,
                    "rrf_score": round(rrf_scores[chunk_idx], 6),
                    "metrics": {
                        "best_dense_rank": trace["best_dense_rank"],
                        "best_sparse_rank": trace["best_sparse_rank"],
                        "min_l2_distance": trace["min_l2_distance"],
                        "max_bm25_score": trace["max_bm25_score"],
                        "dense_hits": trace["dense_hits"],
                        "sparse_hits": trace["sparse_hits"],
                        "matched_query_count": len(trace["matched_queries"]),
                    },
                }
            )

        return reranked_results, per_query_diagnostics

    # --------------------------------------------------------------------------
    # [5단계: 검색된 상위 문서 기반 LLM 답변 생성 (출처 표기 포함)]
    # --------------------------------------------------------------------------
    def generate_answer(self, original_query: str, reranked_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """RRF로 재정렬된 상위 문서 컨텍스트만을 근거로 최종 답변 및 출처 인용을 생성합니다."""
        if not reranked_docs:
            return {
                "answer": "검색된 관련 문서가 없어 답변을 생성할 수 없습니다.",
                "generation_mode": "NONE",
                "citations": [],
            }

        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        for doc in reranked_docs:
            ref_tag = f"[출처 {doc['final_rank']}: {doc['source_file']} p.{doc['page_number']} (청크 #{doc['chunk_id']}, RRF={doc['rrf_score']})]"
            context_blocks.append(f"{ref_tag}\n{doc['snippet']}")
            citations.append(
                {
                    "ref_number": doc["final_rank"],
                    "source_file": doc["source_file"],
                    "page_number": doc["page_number"],
                    "chunk_id": doc["chunk_id"],
                    "doc_id": doc["doc_id"],
                    "rrf_score": doc["rrf_score"],
                }
            )

        joined_context = "\n\n".join(context_blocks)

        if self.genai_client is not None:
            try:
                prompt = (
                    "당신은 신뢰할 수 있는 Advanced RAG 시스템의 AI 어시스턴트입니다.\n"
                    "아래 제공된 [검색 문서 컨텍스트]만을 근거로 사용자의 질문에 정확하고 체계적으로 답변하세요.\n"
                    "답변 작성 규칙:\n"
                    "1. 핵심 답변을 먼저 명확하게 요약한 뒤 세부 내용을 설명하세요.\n"
                    "2. 문장 끝에 반드시 근거가 된 출처 번호(예: [출처 1], [출처 2])를 표기하세요.\n"
                    "3. 컨텍스트에 없는 내용은 추측하지 말고 문서에서 확인되지 않는다고 명시하세요.\n\n"
                    f"[검색 문서 컨텍스트]\n{joined_context}\n\n"
                    f"[사용자 질문]\n{original_query}"
                )
                response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                if response and response.text:
                    return {
                        "answer": response.text.strip(),
                        "generation_mode": "Gemini-2.0-Flash (LLM Synthesis)",
                        "citations": citations,
                    }
            except Exception:
                pass

        # 로컬 추출형 요약 답변 생성 (API 키 미설정 시에도 출처가 명시된 구조화 답변 제공)
        summary_lines = [
            f"질문하신 **\"{original_query}\"**에 대해 RRF(상호 순위 융합)로 선별된 상위 {len(reranked_docs)}개 문서를 분석한 종합 답변입니다.\n"
        ]
        for doc in reranked_docs[:3]:
            summary_lines.append(
                f"- **핵심 근거 {doc['final_rank']}** ({doc['source_file']} {doc['page_number']}페이지, "
                f"RRF 점수 `{doc['rrf_score']}`): {doc['snippet']} **[출처 {doc['final_rank']}]**"
            )
        summary_lines.append(
            "\n> 참고: 현재 로컬 추출형 요약 모드로 동작 중입니다. Cloud Run 또는 서버 환경변수에 `GEMINI_API_KEY`를 설정하면 생성형 LLM 요약이 활성화됩니다."
        )

        return {
            "answer": "\n".join(summary_lines),
            "generation_mode": "Extractive RAG Synthesis (Local Fallback)",
            "citations": citations,
        }


# ==============================================================================
# 4. FastAPI 애플리케이션 및 엔드포인트 구성
# ==============================================================================
app = FastAPI(
    title="Advanced RAG Server (FAISS + Multi-Query + Hybrid Search + RRF)",
    description="FAISS Dense Index, BM25 Sparse Search, Multi-Query Expansion, Reciprocal Rank Fusion 백엔드 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = AdvancedRAGEngine(embedding_dim=768)


def build_error_response(
    status_code: int,
    error_code: str,
    message: str,
    stage: str,
    detail: Optional[str] = None,
) -> JSONResponse:
    """클라이언트(GAS / index.html)에서 정확히 원인을 표시할 수 있는 표준화된 에러 응답"""
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ERROR",
            "error": {
                "code": error_code,
                "stage": stage,
                "message": message,
                "detail": detail or message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        },
    )


@app.get("/health")
@app.get("/status")
async def get_server_status() -> Dict[str, Any]:
    """서버 연동 상태, 인덱스 적재 현황 및 알고리즘 복잡도 메타데이터 반환"""
    return {
        "status": "ONLINE",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vector_db_ready": rag_engine.faiss_index is not None and len(rag_engine.chunks) > 0,
        "index_type": rag_engine.index_type if rag_engine.faiss_index is not None else None,
        "total_chunks": len(rag_engine.chunks),
        "embedding_dim": rag_engine.embedding_dim,
        "loaded_files": rag_engine.loaded_files,
        "llm_configured": rag_engine.genai_client is not None,
        "complexity_info": {
            "chunking": "Time: O(L), Space: O(N * C)",
            "faiss_flat_l2": "Build: O(N * d), Search: O(N * d), Space: O(N * d)",
            "faiss_hnsw": "Build: O(N * d * log N), Search: O(d * log N), Space: O(N * (d + M))",
            "bm25_sparse": "Build: O(N * T_avg), Search: O(Q_t * N_postings)",
            "multi_query_rrf": "Search: O(M_q * (T_dense + T_bm25)), RRF Sort: O(M_q * K * log(M_q * K))",
        },
    }


@app.post("/ingest")
async def ingest_pdf_base64(req: IngestBase64Request):
    """
    [GAS 연동용 PDF 업로드 및 벡터 DB(FAISS + BM25) 구축 엔드포인트]
    Base64로 인코딩된 PDF를 받아 텍스트 추출 -> 청킹 -> 임베딩 -> FAISS & BM25 인덱싱을 수행하며,
    StreamHandler의 실시간 단계별 진행 이벤트를 함께 반환합니다.
    """
    stream_handler = StreamHandler()
    try:
        stream_handler.emit(
            stage="RECEIVING_FILE",
            progress=10,
            message=f"PDF 파일('{req.filename}') 수신 및 Base64 디코딩 중...",
            complexity="O(B) — B: 파일 바이트 크기",
        )
        # Data URL prefix 제거 처리
        b64_clean = req.file_base64.split(",")[-1] if "," in req.file_base64 else req.file_base64
        pdf_bytes = base64.b64decode(b64_clean)

        stream_handler.emit(
            stage="PDF_PARSING_AND_CHUNKING",
            progress=25,
            message=f"'{req.filename}' 텍스트 추출 및 슬라이딩 윈도우 청킹(size={req.chunk_size}, overlap={req.chunk_overlap}) 수행 중...",
            complexity="O(L) — L: 전체 텍스트 길이",
        )
        chunks, total_pages = rag_engine.extract_and_chunk_pdf(
            pdf_bytes=pdf_bytes,
            filename=req.filename,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )

        file_meta = rag_engine.build_indices(
            new_chunks=chunks,
            filename=req.filename,
            total_pages=total_pages,
            index_type=req.index_type,
            stream_handler=stream_handler,
        )

        return {
            "status": "SUCCESS",
            "message": f"'{req.filename}' 기반 벡터 DB(FAISS + BM25) 생성이 완료되었습니다.",
            "file_info": file_meta,
            "stream_events": stream_handler.events,
            "complexity": {
                "faiss_build": (
                    "O(N * d * log N) [IndexHNSWFlat]"
                    if file_meta["index_type"] == "IndexHNSWFlat"
                    else "O(N * d) [IndexFlatL2]"
                ),
                "bm25_build": "O(N * T_avg)",
                "space_complexity": f"O(N * d) — N={file_meta['chunks']}, d={file_meta['embedding_dim']}",
            },
        }
    except ValueError as ve:
        return build_error_response(400, "INVALID_PDF_CONTENT", str(ve), "PDF_PARSING_AND_CHUNKING")
    except Exception as exc:
        return build_error_response(
            500,
            "INGEST_PIPELINE_ERROR",
            f"벡터 DB 생성 중 서버 오류가 발생했습니다: {str(exc)}",
            "FAISS_BM25_INDEXING",
            traceback.format_exc(),
        )


@app.post("/ingest/stream")
async def ingest_pdf_stream(req: IngestBase64Request):
    """
    [실시간 NDJSON 스트리밍 엔드포인트]
    각 진행 단계마다 즉시 NDJSON 청크를 flush하여 클라이언트의 StreamHandler가 실시간 수신하도록 지원합니다.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        stream_handler = StreamHandler()
        try:
            ev1 = stream_handler.emit(
                stage="RECEIVING_FILE",
                progress=10,
                message=f"PDF 파일('{req.filename}') 수신 및 디코딩 중...",
                complexity="O(B)",
            )
            yield stream_handler.format_ndjson_line(ev1)

            b64_clean = req.file_base64.split(",")[-1] if "," in req.file_base64 else req.file_base64
            pdf_bytes = base64.b64decode(b64_clean)

            ev2 = stream_handler.emit(
                stage="PDF_PARSING_AND_CHUNKING",
                progress=30,
                message=f"'{req.filename}' 페이지 파싱 및 청킹 중...",
                complexity="O(L)",
            )
            yield stream_handler.format_ndjson_line(ev2)

            chunks, total_pages = rag_engine.extract_and_chunk_pdf(
                pdf_bytes=pdf_bytes,
                filename=req.filename,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )

            file_meta = rag_engine.build_indices(
                new_chunks=chunks,
                filename=req.filename,
                total_pages=total_pages,
                index_type=req.index_type,
                stream_handler=stream_handler,
            )
            for ev in stream_handler.events[2:]:
                yield stream_handler.format_ndjson_line(ev)

            final_payload = {
                "type": "final_result",
                "status": "SUCCESS",
                "file_info": file_meta,
            }
            yield stream_handler.format_ndjson_line(final_payload)
        except Exception as exc:
            err_event = {
                "type": "error",
                "status": "ERROR",
                "error": {
                    "code": "STREAM_INGEST_ERROR",
                    "message": str(exc),
                },
            }
            yield json.dumps(err_event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/query")
async def query_rag_fusion(req: QueryRequest):
    """
    [Multi-Query + Hybrid Search(FAISS + BM25) + RRF Fusion + LLM 답변 생성 엔드포인트]
    1) 원본 질의를 다중 관점 쿼리로 확장 (Multi-Query Layer)
    2) 각 쿼리마다 FAISS Dense 검색 + BM25 Sparse 검색 병행 (Hybrid Search Layer)
    3) RRF 공식으로 순위 융합 및 상위 K개 문서 재정렬 (RAG Fusion Layer)
    4) 선별된 문서 컨텍스트 기반 LLM 답변 및 출처 표기 생성
    """
    stream_handler = StreamHandler()

    if rag_engine.faiss_index is None or not rag_engine.chunks:
        return build_error_response(
            400,
            "VECTOR_DB_NOT_INITIALIZED",
            "벡터 DB가 아직 생성되지 않았습니다. 먼저 PDF 파일을 선택하고 [벡터 DB 생성] 버튼을 눌러주세요.",
            "PRE_CHECK",
        )

    try:
        # Step 1: Multi-Query 확장
        stream_handler.emit(
            stage="MULTI_QUERY_EXPANSION",
            progress=25,
            message=f"단일 질의를 {req.num_queries}개의 다중 관점 파생 쿼리로 확장 중...",
            complexity=f"O(M_q) — M_q = {req.num_queries + 1}개 쿼리",
        )
        expanded_queries = rag_engine.generate_multi_queries(req.query, num_queries=req.num_queries)

        # Step 2 & 3: Hybrid Search (FAISS + BM25) & RRF Fusion
        stream_handler.emit(
            stage="HYBRID_SEARCH_AND_RRF",
            progress=65,
            message=f"{len(expanded_queries)}개 쿼리에 대해 FAISS({rag_engine.index_type}) + BM25 하이브리드 검색 및 RRF(k={req.rrf_k}) 순위 융합 수행 중...",
            complexity=(
                f"Hybrid Search: O(M_q * (d log N + Q_t * N)), "
                f"RRF Re-ranking: O(M_q * K log(M_q * K))"
            ),
        )
        reranked_docs, query_diagnostics = rag_engine.rag_fusion_rrf(
            expanded_queries=expanded_queries,
            top_k=req.top_k,
            rrf_k=req.rrf_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
        )

        # Step 4: LLM 최종 답변 생성
        stream_handler.emit(
            stage="LLM_ANSWER_GENERATION",
            progress=90,
            message="RRF 상위 문서 컨텍스트를 바탕으로 출처 기반 최종 답변 생성 중...",
            complexity="O(K * C + T_gen)",
        )
        synthesis = rag_engine.generate_answer(req.query, reranked_docs)

        stream_handler.emit(
            stage="COMPLETED",
            progress=100,
            message="Multi-Query + Hybrid Search + RRF 파이프라인 검색 및 답변 생성 완료",
            status="SUCCESS",
        )

        total_elapsed_ms = round((time.perf_counter() - stream_handler.start_time) * 1000, 2)
        n_chunks = len(rag_engine.chunks)
        dim = rag_engine.embedding_dim

        return {
            "status": "SUCCESS",
            "original_query": req.query,
            "expanded_queries": expanded_queries,
            "query_diagnostics": query_diagnostics,
            "retrieved_documents": reranked_docs,
            "llm_answer": synthesis["answer"],
            "generation_mode": synthesis["generation_mode"],
            "citations": synthesis["citations"],
            "stream_events": stream_handler.events,
            "metadata": {
                "index_type": rag_engine.index_type,
                "total_indexed_chunks": n_chunks,
                "embedding_dim": dim,
                "top_k": req.top_k,
                "rrf_k": req.rrf_k,
                "dense_weight": req.dense_weight,
                "sparse_weight": req.sparse_weight,
                "elapsed_ms": total_elapsed_ms,
                "complexity_analysis": {
                    "multi_query_layer": f"O(M_q) where M_q={len(expanded_queries)}",
                    "faiss_dense_layer": (
                        f"O(M_q * d * log N) [IndexHNSWFlat, N={n_chunks}, d={dim}]"
                        if rag_engine.index_type == "IndexHNSWFlat"
                        else f"O(M_q * N * d) [IndexFlatL2, N={n_chunks}, d={dim}]"
                    ),
                    "bm25_sparse_layer": f"O(M_q * Q_t * N) [BM25Okapi, N={n_chunks}]",
                    "rrf_fusion_layer": f"O(M_q * K * log(M_q * K)) [RRF formula: sum 1/(k+rank), k={req.rrf_k}]",
                    "space_complexity": f"O(N * d + V) [N={n_chunks}, d={dim}]",
                },
            },
        }
    except Exception as exc:
        return build_error_response(
            500,
            "QUERY_PIPELINE_ERROR",
            f"질의 처리 중 서버 오류가 발생했습니다: {str(exc)}",
            "RAG_FUSION_PIPELINE",
            traceback.format_exc(),
        )

