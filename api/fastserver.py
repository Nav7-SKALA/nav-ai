import sys
import os
# 현재 디렉토리 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import BASE_DIR, AGENT_ROOT, AGENT_DIR, VECTOR_STORE_ROOT,DB_DIR


sys.path.append(BASE_DIR)
sys.path.append(AGENT_ROOT)
sys.path.append(AGENT_DIR["main_chatbot"])
sys.path.append(AGENT_DIR["mentor_chat"])
sys.path.append(VECTOR_STORE_ROOT)
sys.path.append(DB_DIR)


# 경로 설정 후 import
from typing import Any, Optional, List, Union
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
# from main_chatbot.graph import create_workflow, create_response
from upsert_profile_vector import add_profile_to_vectordb
from career_summary_agent import careerSummary_invoke
from career_title_agent import CareerTitle_invoke
from mentor_chat.mentor_chat_agent import chat_with_mentor

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
    user_query: str = Field(..., example="백엔드 개발 관련 커리어 패스를 알고 싶어요")
    session_id: str = Field(..., example="73b1065c-5850-4602-b935-f45f94f961af")
    user_id: str = Field(..., example="EMP-100014")


class ProfileRequest(BaseModel):
    """
    /apis/v1/career-summary 요청 바디 스키마
    """
    user_id: str = Field(..., example="1")
    user_info: dict = Field(..., example={
        "profileId": 1,
        "years": 2
    })
    projects: list = Field(..., example=[
        {
            "projectId": 1,
            "projectName": "물류 시스템 리뉴얼",
            "projectDescribe": "기존 WMS 시스템 개선 프로젝트",
            "startYear": 1,
            "endYear": 2,
            "projectSize": "대형",
            "skillSets": ["Python", "Django"],
            "roles": ["백엔드 개발자"]
        }
    ])
    certifications: list = Field(..., example=[
        {"name": "정보처리기사", "acquisitionDate": "2023-02"},
        {"name": "AWS SAA", "acquisitionDate": "2023-09"}
    ])
    experiences: list = Field(..., example=[
        {
            "experienceName": "Spring Boot 교육 수료",
            "experienceDescribe": "5주간 백엔드 교육 과정",
            "experiencedAt": "2022-06"
        }
    ])

class RoleModelRequest(BaseModel):
    user_id: str = Field(..., example="testId123")
    input_query: str = Field(..., example="백엔드 개발 전문가가 되는 과정 중에 어떤 게 가장 힘드셨나요?")
    session_id: Optional[str] = Field(..., example="sessionID123") # mongoDB sessionID (롤모델 저장 되어 있는)
    rolemodel_id: str = Field(..., example="6863baadfefc0f239caad583") # rolemodel_id

class RoleModelResponse(BaseModel):
    user_id: str
    chat_summary: str
    answer: str
    success: bool
    error: Optional[str]

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
async def career_path(request: CareerPathRequest):
    """
    메인 챗봇 워크플로우를 실행하여 결과 반환
    """
    try:
        result_state = await run_mainchatbot(request.user_id,request.user_query,request.session_id)
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
        response_data = {
        "user_id": request.user_id,
        "type": None,
        "chat_summary": None,
        "result": None,
        "success": False,
        "error": str(e)
        }
        return JSONResponse(content=response_data)

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

@app.post("/apis/v1/career-summary")
async def process_profile(request: ProfileRequest):
   """백엔드 데이터 받아서 VectorDB 저장 + Career Summary 생성"""
   backend_data = request.dict()
   
   # 1. VectorDB에 저장
   vector_result = add_profile_to_vectordb(backend_data)
   
   # 2. Career Summary 생성
   summary_result = careerSummary_invoke(backend_data)
   
   # 3. 결과 반환
   return {
       "status": "success",
       "profile_id": summary_result["profile_id"],
       "career_summary": summary_result["career_summary"],
       "vector_saved": vector_result
   }

@app.post("/apis/v1/career-title")
def career_title(request: ProfileRequest):
    "Career Title 생성"
    backend_data = request.dict()

    # Career Title 생성
    title_result = CareerTitle_invoke(backend_data)

    # 결과 반환
    return {
       "status": "success",
       "profile_id": title_result["profile_id"],
       "career_title": title_result["career_title"],
   }

@app.post("/apis/v1/rolemodel", response_model=RoleModelResponse)
async def rolemodel_chat(request: RoleModelRequest):
    try:
        result = chat_with_mentor(
            user_id=request.user_id,
            input_query=request.input_query,
            session_id=request.session_id,
            rolemodel_id=request.rolemodel_id  # 이 필드도 추가 필요
        )
        
        return RoleModelResponse(**result)
    except Exception as e:
        return RoleModelResponse(
            user_id=request.user_id,
            chat_summary="",
            answer="오류가 발생했습니다: " + str(e),
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)