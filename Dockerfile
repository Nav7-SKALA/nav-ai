# 1. 베이스 이미지
FROM python:3.11.9-slim

# 2. 작업 디렉토리
WORKDIR /api

# 3. 시스템 패키지 설치 (curl 만 있으면 충분)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 4. Poetry 설치
RUN pip install --upgrade pip poetry

# 5. 의존성 정의 복사 & 설치
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-interaction --no-ansi

# 6. 애플리케이션 코드 복사
COPY . .

# 7. 포트 노출
EXPOSE 8000

#    만약 그냥 스크립트로 띄우려면:
CMD ["python", "fastsever.py"]
