FROM python:3.11.9-slim

WORKDIR /api

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry --break-system-packages

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-interaction --no-ansi

COPY . .

EXPOSE 8000

CMD ["python", "fastserver.py"]