import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import MODEL_NAME, TEMPERATURE
from prompt import careerSummary_prompt



# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

llm = ChatOpenAI(model=MODEL_NAME, api_key=api_key, temperature=TEMPERATURE)


# CareerSummary 프롬프트
cs_prompt = PromptTemplate(
    input_variables=["messages", "information"],
    template=careerSummary_prompt
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
