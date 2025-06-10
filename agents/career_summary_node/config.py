import os

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_PATH=os.path.join(BASE_DIR, 'vector_store')

# 모델 파라미터
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0

