import sys
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from openai import (APIError,
                        AuthenticationError,
                        APITimeoutError,
                        RateLimitError,
                        APIConnectionError
                        )

from config import BASE_DIR, AGENT_ROOT, AGENT_DIR

sys.path.append(BASE_DIR)
sys.path.append(AGENT_ROOT)
sys.path.append(AGENT_DIR['main_chatbot'])
from main_chatbot.graph import create_workflow, create_initial_state, create_response

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
    
    #result = run_main_chatbot(request.user_query)
    graph = create_workflow()
    initial_state = create_initial_state(request.user_query)

    try:
        result = graph.invoke(initial_state)
        result_content = create_response(result)
        
        return result_content

    except APITimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"요청 제한 시간 초과: {e.__class__.__name__}"
        )
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=f"API 요청 한도 초과: {e.__class__.__name__}"
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail=f"인증 오류: {e.__class__.__name__}"
        )
    except APIConnectionError as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI 연결 오류: {e.__class__.__name__}"
        )
    except APIError as e:
        # 그 외 OpenAI 관련 에러: 서버 오류(500)
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API 오류: {e.__class__.__name__}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류: {str(e)}"
        )


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