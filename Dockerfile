# 1. 베이스 이미지
FROM python:3.11.9-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 시스템 패키지(필요 시 추가)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 4. Poetry 설치
RUN pip install --upgrade pip poetry

# 5. 의존성 정의 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-chache -r requirements.txt

# 6. 애플리케이션 코드 복사
COPY . /app

# 7. 컨테이너가 열 포트
EXPOSE 8000

# 8. 서버 실행 커맨드
CMD ["python","fastsever.py"]
