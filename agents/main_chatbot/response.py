from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PromptWrite(BaseModel):
    """re-write prompt"""
    new_query: str = Field(
        description="재생성된 사용자 질의"
    )
    reason: str = Field(
        description="그렇게 수정하게 된 이유 설명"
    )

class PathRecommendResult(BaseModel):
    career_path_text: str = Field(
        description="커리어 증진 경로 추천 텍스트 자료"
    )
    career_path_roadmap: Optional[Dict[str, Any]] = Field(
        default=None,
        description="커리어 증진 경로 추천 로드맵"
    )

## role-model 관련 내부 클래스들(출력 형식)
class RoleModelProfile(BaseModel):
    """단일 롤모델 프로필"""
    name: str = Field(description="가상 롤모델의 이름")
    current_position: str = Field(description="현재 직무/포지션")
    experience_years: str = Field(description="경력 년차")
    main_domains: List[str] = Field(description="주요 도메인 리스트")
    tech_stack: List[str] = Field(description="핵심 기술스택 리스트")
    career_highlights: List[str] = Field(description="주요 경력 하이라이트")
    advice_message: str = Field(description="사용자에게 주는 조언 메시지")

class RoleModelGroup(BaseModel):
    """롤모델 그룹 정보"""
    group_name: str = Field(description="그룹명 (예: 백엔드 전문가, 풀스택 개발자)")
    group_description: str = Field(description="그룹의 특징과 공통점 설명")
    member_count: int = Field(description="그룹에 속한 사원 수")
    common_tech_stack: List[str] = Field(description="공통 기술 스택")
    common_career_path: List[str] = Field(description="공통 커리어 패턴")
    role_model: RoleModelProfile = Field(description="그룹 대표 롤모델")
    real_info: List[str] = Field(description="그룹에 속한 사원의 chromaDB ID")

class GroupedRoleModelResult(BaseModel):
    """그룹화된 롤모델 결과"""
    analysis_summary: str = Field(description="전체 분석 요약")
    total_employees: int = Field(description="분석한 총 사원 수")
    groups: List[RoleModelGroup] = Field(description="생성된 롤모델 그룹들 (2-4개)")