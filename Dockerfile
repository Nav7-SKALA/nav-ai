FROM python:3.11.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app:/app/api:/app/agents:/app/agents/main_chatbot:/app/tools

COPY api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api/fastserver.py"]
