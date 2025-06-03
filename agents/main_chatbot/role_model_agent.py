import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import roleModel_prompt
from config import MODEL_NAME, TEMPERATURE

# 환경변수 로드
load_dotenv()


# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

roleModel_chain = PromptTemplate.from_template(roleModel_prompt) | llm | StrOutputParser()

def roleModel_invoke(state: dict, config=None) -> dict:
    """roleModel Chain 실행 함수"""

    messages_text = "\n".join([
        msg.content for msg in state.get("messages", [])
        if hasattr(msg, 'content')
    ])

    # 사용자 정보
    information = state.get("information", "")

    result = roleModel_chain.invoke({
        "messages": messages_text,
        "information": information
    })

    new_messages = [] #list(state.get("messages", []))
    new_messages.append(AIMessage(content=result, name="RoleModel"))

    return {
        **state,
        "messages": new_messages
    }

def roleModel_node(state):
    """RoleModel 노드 함수"""
    return roleModel_invoke(state)

if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="cloud PM 롤모델 찾아줘")]}
    result = roleModel_invoke(test_state)
    print(result["messages"][-1].content)