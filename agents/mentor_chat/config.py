import os

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

AGENT_ROOT = os.path.join(BASE_DIR, "agents")
AGENT_DIR = {'main_chatbot': os.path.join(AGENT_ROOT, "main_chatbot"),
             'mentor_chatbot': os.path.join(AGENT_ROOT, "mentor_chatbot"),
             'summary': os.path.join(AGENT_ROOT, "summary"),
            }

DB_DIR = os.path.join(BASE_DIR, "db")