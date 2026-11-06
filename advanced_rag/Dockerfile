# ==============================================================================
# Dockerfile for Google Cloud Run Deployment
# Service: Advanced RAG (FAISS + Multi-Query + Hybrid Search + RRF Fusion)
# ==============================================================================
FROM python:3.11-slim

# 시스템 패키지 설치 (FAISS 구동에 필요한 libgomp1 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY main.py .

# Cloud Run 기본 포트 환경변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Uvicorn 서버 실행
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

