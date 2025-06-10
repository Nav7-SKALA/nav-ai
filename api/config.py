import os

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

AGENT_ROOT = os.path.join(BASE_DIR, "agents")
AGENT_DIR = {'main_chatbot': os.path.join(AGENT_ROOT, "main_chatbot"),
             'persona_chat': os.path.join(AGENT_ROOT, "persona_chat"),
             'summary': os.path.join(AGENT_ROOT, "summary"),
             'career_summary': os.path.join(AGENT_ROOT, "career_summary")
            }

# 모델 파라미터
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0