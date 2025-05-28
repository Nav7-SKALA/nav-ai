FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry --break-system-packages

COPY 

COPY pyproject.toml poetry.lock ./

    poetry install --only=main --no-interaction --no-ansi --no-root

COPY . .

EXPOSE 8000

CMD ["python", "api/fastsever.py"]