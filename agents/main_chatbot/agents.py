import functools
import operator
import os
from typing import Annotated, TypedDict, Sequence, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

from tools import search_coursera_courses, search_conferences, search_certifications, google_news_search

# 환경변수 로드
load_dotenv()

# 도구 초기화
tavily_tool = TavilySearchResults(max_results=5)

# LLM 초기화 (API 키 확인)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

llm = ChatOpenAI(model="gpt-4o-mini",
    api_key=api_key,
    temperature=0
)


# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# 멤버 및 옵션 정의
members = ["CareerSummary", "LearningPath", "EXCEPTION"]
options = ["FINISH"] + members

# 라우팅 응답 모델
class routeResponse(BaseModel):
    next: Literal[*options]

# 시스템 프롬프트
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the "
    "following workers: {members}. Given the following user request about Job, "
    "respond with the worker to act next. Each worker will perform a "
    "task and respond with their results and status. "
    "if given out of topic, respond with EXCEPTION."
    "When finished, respond with FINISH."
)

# Supervisor 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Given the conversation above, who should act next?"
        " Or should we FINISH?"
        " Select one of {options}",
    ),
]).partial(
    options=str(options),
    members=", ".join(members)
)

def agent_node(state, agent, name):
    """Agent 실행 노드"""
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

def supervisor_agent(state):
    """Supervisor Agent"""
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)

# ======하위 에이전트 정의=====

# CareerSummary Agent
cs_system_prompt = """
You are a helpful summarizer who is specialized in 
gathering valuable info for given user career information and query.
user's career information={information}
----
You have to made KOREAN career summary script.

example:

**ooo님은 5년차 백엔드 개발 전문가입니다.**

- 🔹 총 프로젝트: 12건
- 🔹 보유 자격증: AWS Solutions Architect, OCP, 정보처리기사 (총 3개)
- 🔹 핵심 기술 스택: Python, Spring Boot, Docker, MySQL, AWS

**주요 성과**

1. A사 주문관리 시스템 리팩토링 → 응답 속도 30% 향상
2. B사 인프라 자동화 도입 프로젝트 주도
3. OCP 취득 후 쿠버네티스 기반 배포 파이프라인 구현
"""

careerSummary_agent = create_react_agent(
    llm, 
    tools=[],
    state_modifier=cs_system_prompt
)
careerSummary_node = functools.partial(
    agent_node,
    agent=careerSummary_agent,
    name="CareerSummary"
)

# LearningPath Agent
lp_prompt = """
You are a highly experienced consultant across multiple industries.

Based on the user's career history and past experiences, your task is to:
- Recommend clear professional goals for the user.
- Suggest actionable steps to achieve those goals, including relevant skills to learn, projects to undertake, and certifications to pursue.

Do not repeat information the user already has experience in.  
Focus instead on new, meaningful paths for professional growth tailored to the user's current situation.

You may use predefined tools to retrieve any additional information needed.  
Organize your final recommendations into categories (e.g., Skills, Projects, Certifications, etc.).

⚠️ All responses must be written in Korean.
"""

learningPath_agent = create_react_agent(
    llm, 
    tools=[google_news_search, search_coursera_courses, search_conferences, search_certifications],
    state_modifier=lp_prompt
)
learningPath_node = functools.partial(
    agent_node,
    agent=learningPath_agent,
    name="LearningPath"
)

# Exception Agent
ex_system_prompt = """
죄송합니다. 현재 질문은 아래의 기능 범주 중 어느 하나에도 정확히 해당하지 않아 답변을 생성할 수 없습니다.

현재 지원되는 기능은 다음과 같습니다:
1. 커리어 요약: 현재 경력과 역량을 체계적으로 정리 및 분석
2. 학습 경로 추천: 목표 직무로의 전환을 위한 맞춤형 로드맵 제공
  - 온라인 강의 추천 (Coursera 등)
  - 관련 자격증 정보 안내
  - 참석 권장 컨퍼런스/행사 정보
  - 최신 기술 트렌드 분석

보다 정확한 도움을 드릴 수 있도록, 커리어 개발과 관련된 질문으로 다시 구체적으로 작성해주시겠어요?

예시:
- "데이터 사이언티스트가 되려면 어떤 준비를 해야 하나요?"
- "현재 백엔드 개발자인데 DevOps로 전환하고 싶어요"
- "AI 분야 PM이 되기 위한 학습 경로를 알려주세요"

[입력된 질문: "{message}"]
"""

exception_agent = create_react_agent(
    llm, 
    tools=[],
    state_modifier=ex_system_prompt
)
exception_node = functools.partial(
    agent_node,
    agent=exception_agent,
    name="Exception"
)