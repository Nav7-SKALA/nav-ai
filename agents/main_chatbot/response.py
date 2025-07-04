from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class PromptWrite(BaseModel):
    """re-write prompt"""
    new_query: str = Field(
        description="재생성된 사용자 질의"
    )
    reason: str = Field(
        description="그렇게 수정하게 된 이유 설명"
    )

## roadmap 관련 내부 클래스들 (출력 형식)
class RecommendRoadmap(BaseModel):
    """경력 증진 추천 로드맵(프로젝트)"""
    period: str = Field(description="수행 프로젝트 기간")
    project: str = Field(description="수행 프로젝트명")
    role: str = Field(description="프로젝트 내 수행 역할")
    job: str = Field(description="프로젝트 내 수행 직무")
    key_skills: str = Field(description="프로젝트 수행 시 활용 주요 기술 스택")
    growth_focus: str = Field(description="해당 수행 프로젝트이 경력 증진에 기여하는 점")

class SimilarRoadmapProject(BaseModel):
    """유사 구성원 경력 로드맵 (프로젝트)"""
    period: str = Field(description="수행 프로젝트 기간")
    name: str = Field(description="수행 프로젝트명")
    role: str = Field(description="프로젝트 내 수행 역할")
    job: str = Field(description="프로젝트 내 수행 직무")
    detail: str = Field(description="프로젝트 상세 내용")

class SimilarRoadmapExperience(BaseModel):
    """유사 구성원 경력 로드맵 (경험)"""
    name: str = Field(description="경력 증진 관련 경험명 (예: 세미나, 컨퍼런스, 해커톤 등)")

class SimilarRoadmapCertification(BaseModel):
    """유사 구성원 경력 로드맵 (자격증)"""
    name: str = Field(description="경력 증진 위해 취득한 실제 자격증명")


class SimilarRoadmapProjectBlock(BaseModel):
    """프로젝트 블록"""
    project: List[SimilarRoadmapProject]

class SimilarRoadmapExperienceBlock(BaseModel):
    """경험 블록"""
    experience: List[SimilarRoadmapExperience] = Field(default=[])

class SimilarRoadmapCertificationBlock(BaseModel):
    """자격증 블록"""
    certification: List[SimilarRoadmapCertification] = Field(default=[])


import re
def format_text_with_newlines(text: str) -> str:
    """문장이 끝나면 개행 문자를 넣어서 포맷팅"""
    if not text:
        return text
        
    # 문장 끝 패턴: 마침표, 물음표, 느낌표 뒤에 공백이나 문자가 오는 경우
    # 단, 숫자.숫자 같은 소수점은 제외
    sentence_end_pattern = r'([.!?])(\s+)(?=[A-Za-z가-힣])'
        
    # 문장 끝에 개행 문자 추가
    formatted_text = re.sub(sentence_end_pattern, r'\1\n\n', text)
        
    # 연속된 개행 문자 정리 (최대 2개까지만)
    # formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        
    return formatted_text.strip()


class SimilarRoadMapResult(BaseModel):
    similar_analysis_text: str = Field(
        default="유사한 경력 경로를 가진 사내 구성원의 성장 데이터를 찾을 수 없습니다.\n원하는 경력 목표 직무를 더 자세히 작성해주세요.",
        description="유사 구성원의 경력 증진 로드맵 제시 전 설명 텍스트"
    )
    # similar_analysis_roadmap: List[Union[SimilarRoadmapProjectBlock,
    #                                      SimilarRoadmapExperienceBlock,
    #                                      SimilarRoadmapCertificationBlock]] = Field(
    #     default=[],
    #     description="유사 구성원의 경력 증진 로드맵 (항목별 블록 리스트)"
    # )
    project: SimilarRoadmapProjectBlock = Field(
        default_factory=SimilarRoadmapProjectBlock,
        description="유사 구성원의 경력 증진 프로젝트 블록"
    )
    experience: SimilarRoadmapExperienceBlock = Field(
        default_factory=SimilarRoadmapExperienceBlock,
        description="유사 구성원의 경력 증진 경험 블록"
    )
    certification: SimilarRoadmapCertificationBlock = Field(
        default_factory=SimilarRoadmapCertificationBlock,
        description="유사 구성원의 경력 증진 자격증 블록"
    )

    def to_output_dict(self):
        return {
            'similar_text': format_text_with_newlines(self.similar_analysis_text),
            # 'similar_roadmaps': [r.model_dump() for r in self.similar_analysis_roadmap]
            'similar_roadmaps': [
                self.project.model_dump(),
                self.experience.model_dump(),
                self.certification.model_dump()
            ]
        }

class PathRecommendResult(BaseModel):
    career_path_text: str = Field(
        default="입력 정보가 부족하거나 분석 결과가 제한적이므로 경력 증진 경로를 제공하기 어렵습니다.\n향후 직무 추천을 원한다면, '향후에는 어떤 직무를 갖게 될까?'와 같은 질의로 다시 시도해주세요.",
        description="커리어 증진 경로 추천 텍스트 자료"
    )
    career_path_roadmap: List[RecommendRoadmap] = Field(
        default=[],
        description="커리어 증진 경로 추천 로드맵"
    )

    def to_output_dict(self):
        return {
            'text': format_text_with_newlines(self.career_path_text),
            'roadmaps': [r.model_dump() for r in self.career_path_roadmap]
        }


## role-model 관련 내부 클래스들(출력 형식)
class RoleModelProfile(BaseModel):
    """멘토(개인) 프로필"""
    name: str = Field(description="가상 롤모델의 이름")
    current_position: str = Field(description="현재 직무/포지션")
    experience_years: str = Field(description="경력 년차")
    main_domains: List[str] = Field(description="주요 도메인 리스트")
    # skill_set: List[str] = Field(description="핵심 기술 스택 리스트")
    # career_highlights: List[str] = Field(description="주요 경력 하이라이트")
    advice_message: str = Field(description="사용자에게 주는 조언 메시지")

class RoleModelGroup(BaseModel):
    """멘토(그룹) 정보"""
    group_name: str = Field(description="그룹명 (예: 백엔드 전문가, 풀스택 개발자)")
    group_description: str = Field(description="그룹의 특징과 공통점 설명")
    # member_count: int = Field(description="그룹에 속한 사원 수")
    common_skill_set: List[str] = Field(description="공통 기술 스택")
    common_career_path: List[str] = Field(description="공통 커리어 패턴")
    role_model: RoleModelProfile = Field(description="그룹 대표 롤모델")
    real_info: List[Any] = Field(default=[], description="그룹에 속한 사원의 chromaDB ID")
    common_project : List[str] = Field(description="그룹에 속한 사원들의 주요 수행 프로젝트")
    common_experience : List[str] = Field(
        default=[],
        description="그룹에 속한 사원들의 프로젝트 외 경력 관련 경험(예: 컨퍼런스 참여) 정보")
    common_cert : List[str] = Field(
        default=[],
        description="그룹에 속한 사원들의 경력 관련 자격증 정보")

class GroupedRoleModelResult(BaseModel):
    """그룹화된 롤모델 결과"""
    analysis_summary: str = Field(description="전체 분석 요약")
    groups: List[RoleModelGroup] = Field(description="생성된 롤모델 그룹들 (2-4개)")
    analysis_summary: str = Field(
        default='',
        description="전체 분석 요약")
    groups: List[RoleModelGroup] = Field(
        default=[],
        description="생성된 롤모델 그룹들 (3개)")

class LectureRecommendation(BaseModel):
    internal_course: str
    ax_college: str
    explanation: str

class TrendResult(BaseModel):
    text: str
    ax_college: str