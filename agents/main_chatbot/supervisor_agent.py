import operator
import os
from typing import Annotated, TypedDict, Sequence, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

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

# Supervisor 체인
supervisor_chain = prompt | llm.with_structured_output(routeResponse)

def supervisor_agent(state):
    """Supervisor Agent"""
    return supervisor_chain.invoke(state)