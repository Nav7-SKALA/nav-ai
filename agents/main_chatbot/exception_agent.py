import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import exception_prompt
from config import MODEL_NAME, TEMPERATURE

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

ex_prompt = PromptTemplate(
                    input_variables=["messages"],
                    template=exception_prompt
            )

exception_chain = ex_prompt | llm | StrOutputParser()

def exception_invoke(state: dict, config=None) -> dict:
    """exception Chain 실행 함수"""

    last_message = ""
    if state.get("messages"):
        for msg in reversed(state.get("messages")):
            if hasattr(msg, "content"):
                last_message = msg.content
                break

    result = exception_chain.invoke({
        "messages": last_message,
    })

    new_messages = [] #list(state.get("messages", []))
    new_messages.append(AIMessage(content=result))

    return {
        **state,
        "messages": new_messages
    }

def exception_node(state):
    """Exception 노드 함수"""
    return exception_invoke(state)

if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="나 저녁 뭐 먹을깡?")]}
    result = exception_invoke(test_state)
    print(result["messages"][-1].content)