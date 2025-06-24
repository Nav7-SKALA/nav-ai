import sys
from typing import Any, Optional, List, Union
from dotenv import load_dotenv
load_dotenv("api/.env")
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from pydantic import BaseModel, Field
from agents.main_chatbot.cdgraph import run_mainchatbot,create_initial_state
from openai import (
    APIError,
    AuthenticationError,
    # APITimeoutError,
    RateLimitError,
    APIConnectionError,
)
import uvicorn

# 프로젝트 경로 설정
from api.config import BASE_DIR, AGENT_ROOT, AGENT_DIR

sys.path.append(BASE_DIR)
sys.path.append(AGENT_ROOT)
sys.path.append(AGENT_DIR["main_chatbot"])
from main_chatbot.graph import create_workflow, create_initial_state, create_response

app = FastAPI(
    docs_url="/apis/docs",
    openapi_url="/apis/openapi.json",
    title="Nav7 FastAPI",
    version="1.0.0",
)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Pydantic 모델 정의
# ──────────────────────────────────────────────────────────────────────────────

class CareerPathResult(BaseModel):
    """
    result 필드 스키마: 실제 봇 에이전트가 반환하는 내용
    """
    agent: str = Field(
        ..., example="LearningPath", description="사용된 에이전트 이름"
    )
    text: Union[str, List[Any]] = Field(
        ...,
        example=(
            "다음은 클라우드 관련 강의 정보입니다:\n\n"
            "1. **클라우드 컴퓨팅 기초 (Cloud 101)**\n"
            "   - [강의 링크](https://www.coursera.org/learn/cloud-computing-basics-ko)\n\n"
            "이 강의는 클라우드 컴퓨팅의 기본 개념을 배우고 싶으신 분들에게 적합합니다. "
            "추가적인 클라우드 관련 강의가 필요하시면 말씀해 주세요!"
        ),
    )


class SuccessContent(BaseModel):
    """
    content.success / content.result 을 감싸는 구조
    """
    success: bool = Field(..., description="요청 처리 성공 여부")
    result: CareerPathResult = Field(..., description="에이전트 결과 객체")


class CareerPathResponse(BaseModel):
    """
    최종 성공 응답 스키마:
    { "content": { "success": true, "result": {...} } }
    """
    content: SuccessContent = Field(..., description="성공 응답 컨텐츠")


class ErrorResult(BaseModel):
    """
    오류 응답의 result 안에 담길 detail 객체
    """
    detail: str = Field(..., example="ERROR DETAIL MESSAGE", description="에러 상세 메시지")


class ErrorContent(BaseModel):
    """
    content.success = false, content.result에 ErrorResult
    """
    success: bool = Field(False, description="항상 False")
    result: ErrorResult = Field(..., description="오류 상세 정보")


class ErrorResponseModel(BaseModel):
    """
    최종 오류 응답 스키마:
    { "content": { "success": false, "result": { "detail": "에러 상세 메시지" } } }
    """
    content: ErrorContent = Field(..., description="오류 응답 컨텐츠")


class CareerPathRequest(BaseModel):
    """
    /apis/v1/career-path 요청 바디 스키마
    """
    user_query: str = Field(..., example="클라우드 관련 강의 정보 알려줘")
    session_id: str = Field(..., example="session_id")
    user_id: str = Field(..., example="user_12345")
    career_summary: str = Field(...,example="사용자의 커리어 요약 정보")


# ──────────────────────────────────────────────────────────────────────────────
# 2) 전역 예외 핸들러 등록
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """
    요청(Request) 유효성 검사 실패 시(422) ErrorResponseModel 형태로 반환
    """
    first_error = exc.errors()[0]
    loc = ".".join(str(x) for x in first_error.get("loc", []))
    msg = first_error.get("msg", "Validation error")
    detail_msg = f"{loc}: {msg}"
    error_content = ErrorContent(success=False, result=ErrorResult(detail=detail_msg))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponseModel(content=error_content).dict(exclude_none=True),
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(
    request: Request, exc: ResponseValidationError
):
    """
    응답(Response) 유효성 검사 실패 시(500) ErrorResponseModel 형태로 반환
    """
    raw = exc.raw_errors[0]
    msg = raw.exc_msg if hasattr(raw, "exc_msg") else str(raw)
    detail_msg = f"Response validation error: {msg}"
    error_content = ErrorContent(success=False, result=ErrorResult(detail=detail_msg))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponseModel(content=error_content).dict(exclude_none=True),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTPException이 발생했을 때 ErrorResponseModel 형태로 반환
    """
    error_content = ErrorContent(
        success=False,
        result=ErrorResult(detail=exc.detail),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponseModel(content=error_content).dict(exclude_none=True),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    처리되지 않은 예외는 모두 500으로 ErrorResponseModel 형태로 반환
    """
    error_content = ErrorContent(
        success=False,
        result=ErrorResult(detail=str(exc)),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponseModel(content=error_content).dict(exclude_none=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3) /apis/v1/career-path 엔드포인트 정의
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/apis/v1/career-path",
    response_model=CareerPathResponse,
    responses={
        200: {"description": "성공 조회", "model": CareerPathResponse},
        400: {"description": "잘못된 요청 (InvalidRequest)", "model": ErrorResponseModel},
        401: {"description": "인증 실패", "model": ErrorResponseModel},
        422: {"description": "유효성 검사 오류", "model": ErrorResponseModel},
        429: {"description": "요청 한도 초과 (RateLimit)",
             "model": ErrorResponseModel,
             },
        504: {
            "description": "외부 서비스 타임아웃 (Gateway Timeout)",
            "model": ErrorResponseModel,
        },
        500: {"description": "서버 내부 오류", "model": ErrorResponseModel},
    },
    summary="Main chatbot API",
    description=(
        "사용자 query를 받아 메인 챗봇 워크플로우(graph)를 실행한 뒤 "
        "답변을 반환합니다."
    ),
)
def career_path(request: CareerPathRequest):
    """
    메인 챗봇 워크플로우를 실행하여 결과 반환
    """
    try:
        result_state = run_mainchatbot(request.user_id,request.user_query,request.career_summary)
        response_data = {
        "user_id": result_state.get("user_id"),
        "type": result_state.get("intent"),
        "chat_summary": result_state.get("chat_summary"),
        "result": result_state.get("result"),
        "success": True,
        "error": None
        }
        return JSONResponse(content=response_data)

    ## 테스트를 위해 에러 발생 시 하드코딩 결과 입력
    except Exception as e:
        result_content = create_initial_state(request.user_id, request.user_query, request.career_summary)
        result_content["success"]=False
        response_data = {
        "user_id": request.user_id,
        "type": None,
        "chat_summary": None,
        "result": None,
        "success": False,
        "error": str(e)
        }
        return JSONResponse(content=result_content.dict(exclude_none=False))

    ## 원래 에러 핸들링 코드 
    # except APITimeoutError as e:
    #     raise HTTPException(
    #         status_code=504,
    #         detail=f"요청 제한 시간 초과: {str(e)}",
    #     )
    # except RateLimitError as e:
    #     raise HTTPException(
    #         status_code=429,
    #         detail=f"API 요청 한도 초과: {str(e)}",
    #     )
    # except AuthenticationError as e:
    #     raise HTTPException(
    #         status_code=401,
    #         detail=f"인증 오류: {str(e)}",
    #     )
    # except APIConnectionError as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"OpenAI 연결 오류: {str(e)}",
    #     )
    # except APIError as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"OpenAI API 오류: {str(e)}",
    #     )
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"서버 내부 오류: {str(e)}",
    #     )

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