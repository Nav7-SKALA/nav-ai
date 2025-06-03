FROM python:3.11.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app:/app/agents:/app/agents/main_chatbot:/app/agents/tools

COPY api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY api/fastserver.py ./fastserver.py
COPY agents/ ./agents/


EXPOSE 8000

CMD ["python", "fastserver.py"]