import sys
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from config import BASE_DIR, AGENT_ROOT, AGENT_DIR

sys.path.append(BASE_DIR)
sys.path.append(AGENT_ROOT)
sys.path.append(AGENT_DIR['main_chatbot'])
from main_chatbot.graph import run_main_chatbot

app = FastAPI(docs_url="/apis/docs", openapi_url="/apis/openapi.json")

# @app.post("/apis/v1/career-path")
# def career_path():
#     return "커리어추천: cloud 강의 듣기"

# Request 정의
class career_path_request(BaseModel):
    user_query: str
    user_id: str

@app.post("/apis/v1/career-path")
def career_path(request: career_path_request):
    """메인 챗봇 API"""
    
    result = run_main_chatbot(request.user_query)
    
    return result


@app.post("/apis/v1/rolemodel")
def rolemodel():
    return "rolemodel: cloud 마스터"

@app.post("/apis/v1/career-summary")
def career_summary():
    return "career-summary: 당신은 cloud 최고 전문가 김채연매니저입니다>_<"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)