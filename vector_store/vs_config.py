import os

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 파일 경로
HISTORY_PATH = os.path.join(BASE_DIR, "data", "직원history.csv")
SKILLSET_PATH = os.path.join(BASE_DIR, "data", "skillset_data.csv")
DOCS_DIR = os.path.join(BASE_DIR,"data", "emp_docs")

DB_PATH=os.path.join(BASE_DIR, "db")


# 🌐 외부 접속용 (FastAPI가 다른 네임스페이스에 있을 때)
CHROMA_HOST = "chromadb-1.skala25a.project.skala-ai.com"
CHROMA_PORT = 443
USE_SSL = True
AUTH_HEADER = {
    "Authorization": "Basic YWRtaW46U2thbGEyNWEhMjMk"
}
