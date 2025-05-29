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

# Exception 프롬프트
ex_prompt = PromptTemplate(
    input_variables=["message"],
    template="""
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
)

# Exception 체인
exception_chain = ex_prompt | llm | StrOutputParser()


# thread_id 추가해야됨
#저장된 state에서 가장 마지막 메세지값만 꺼내와서 보여주도록 설계함 -> 이렇게 할지 아니면 state["input_query"]와 message를 같이 반환할지 고민중
def exception_invoke(state: dict, config=None) -> dict:
    """Exception Chain 실행 함수"""
    last_message = ""
    if state.get("messages"):
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'content') and msg.content.strip():
                last_message = msg.content
                break
    
    result = exception_chain.invoke({
        "message": last_message
    })
    
    new_messages = list(state.get("messages", []))
    new_messages.append(HumanMessage(content=result, name="Exception"))
    
    return {
        **state,
        "messages": new_messages
    }

def exception_node(state):
    """Exception 노드 함수"""
    return exception_invoke(state)