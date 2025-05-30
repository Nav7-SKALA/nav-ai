import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import careerSummary_prompt
from config import MODEL_NAME, TEMPERATURE

import sys
# 현재 파일의 상위 디렉토리 (nav-ai)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tools.google_news_tool import google_news_search
from tools.vector_search_tool import vectorDB_search
from tools.postgres_tool import RDB_search
from tools.web_search_tool import tavily_tool

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

tools = [RDB_search, vectorDB_search]

# CareerSummary 프롬프트
cs_prompt = PromptTemplate(
    input_variables=["messages"],
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

    result = careerSummary_chain.invoke({
        "messages": messages_text,
        "information": ""
    })
    
    new_messages = [] #list(state.get("messages", []))
    new_messages.append(AIMessage(content=result, name="CareerSummary"))
    
    return {
        **state,
        "messages": new_messages
    }

def careerSummary_node(state):
    """CareerSummary 노드 함수"""
    return careerSummary_invoke(state)


if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="AI PM 되려면 어떻게 해야 해?")]}
    result = careerSummary_invoke(test_state)
    print(result["messages"][-1].content)