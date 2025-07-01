import os

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

AGENT_ROOT = os.path.join(BASE_DIR, "agents")
AGENT_DIR = {'main_chatbot': os.path.join(AGENT_ROOT, "main_chatbot"),
             'persona_chat': os.path.join(AGENT_ROOT, "persona_chat"),
             'summary': os.path.join(AGENT_ROOT, "summary"),
            }
# Vector Store 경로 추가
VECTOR_STORE_ROOT = os.path.join(BASE_DIR, "vector_store")

# 모델 파라미터
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0