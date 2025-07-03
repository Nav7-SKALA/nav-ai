import os
import sys

# 프로젝트 루트 경로 계산 (tools -> agents -> nav-ai)
current_file = os.path.abspath(__file__)              # config.py
tools_dir = os.path.dirname(current_file)             # agents/tools/
agents_dir = os.path.dirname(tools_dir)               # agents/
project_root = os.path.dirname(agents_dir)            # nav-ai/

# 프로젝트 루트를 sys.path에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기본 경로 설정 (프로젝트 루트 기준)
BASE_DIR = project_root
# Vector Store 경로 추가
VECTOR_STORE_ROOT = os.path.join(BASE_DIR, "vector_store")

# 모델 파라미터
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0