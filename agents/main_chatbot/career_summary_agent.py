import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

# CareerSummary 프롬프트
cs_prompt = PromptTemplate(
    input_variables=["messages", "information"],
    template="""
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

Messages: {messages}
"""
)

# CareerSummary 체인
careerSummary_chain = cs_prompt | llm | StrOutputParser()

def careerSummary_invoke(state: dict, config=None) -> dict:
    """CareerSummary Chain 실행 함수"""
    messages_text = "\n".join([
        msg.content for msg in state.get("messages", [])
        if hasattr(msg, 'content')
    ])
    # information = state.get("information", "")  # 사용자 정보가 있다면
    information = "3년차 프론트엔드 개발자"
    
    result = careerSummary_chain.invoke({
        "messages": messages_text,
        "information": information
    })
    
    new_messages = list(state.get("messages", []))
    new_messages.append(HumanMessage(content=result, name="CareerSummary"))
    
    return {
        **state,
        "messages": new_messages
    }

def careerSummary_node(state):
    """CareerSummary 노드 함수"""
    return careerSummary_invoke(state)