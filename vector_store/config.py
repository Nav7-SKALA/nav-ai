import os

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 파일 경로
HISTORY_PATH = os.path.join(BASE_DIR, "data", "직원history.csv")
SKILLSET_PATH = os.path.join(BASE_DIR, "data", "skillset_data.csv")
DOCS_DIR = os.path.join(BASE_DIR,"data", "emp_docs")