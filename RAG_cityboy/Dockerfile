1. 경량화된 Python 3.10 이미지 사용

FROM python:3.10-slim

2. 작업 디렉토리 설정

WORKDIR /app

3. 환경 변수 설정 (파이썬 출력 버퍼링 방지)

ENV PYTHONUNBUFFERED=1

4. 의존성 파일 복사 및 설치

캐시를 사용하지 않아 이미지 용량을 최적화합니다.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

5. 애플리케이션 코드 복사

COPY . .

6. 구글 클라우드 런은 기본적으로 PORT 8080을 사용

ENV PORT 8080

7. Uvicorn을 사용하여 FastAPI 앱 실행 (0.0.0.0으로 바인딩)

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
