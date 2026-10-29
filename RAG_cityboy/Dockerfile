# Cloud Run 배포용 이미지
FROM python:3.12-slim

# 로그가 버퍼링 없이 즉시 출력되도록 설정
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 의존성 먼저 설치 (소스 변경 시 캐시 재사용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Cloud Run 은 PORT 환경 변수로 포트를 전달한다 (기본 8080)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
