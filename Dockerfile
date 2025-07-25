# # FROM python:3.11-slim-bullseye
# FROM python:3.11.9

# WORKDIR /app

# # 시스템 패키지 설치
# # RUN apt-get update && apt-get install -y \
# #     build-essential \
# #     libpq-dev \
# #     gcc \
# #     git \
# #     default-jdk \
# #     curl \
# #     && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends curl && \
#     rm -rf /var/lib/apt/lists/*

# ENV PYTHONPATH=/app:/app/api:/app/agents:/app/agents/main_chatbot:/app/tools

# COPY api/requirements.txt ./requirements.txt

# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# COPY . .

# EXPOSE 8000

# CMD ["python", "api/fastserver.py"]

FROM python:3.11.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

ENV ACCELERATE_DISABLE_RICH=1
ENV TRANSFORMERS_VERBOSITY=error
ENV ACCELERATE_TORCH_DEVICE=cpu
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV PYTHONPATH=/app:/app/api:/app/agents:/app/agents/main_chatbot:/app/tools

COPY api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api/fastserver.py"]
