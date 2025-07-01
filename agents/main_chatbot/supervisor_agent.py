import os
import operator
from typing import Annotated, TypedDict, Sequence, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from prompt import supervisor_prompt
from config import MODEL_NAME, TEMPERATURE

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

members = ["CareerSummary", "LearningPath", "RoleModel", "EXCEPTION"]
options = ["FINISH"] + members


sv_prompt = ChatPromptTemplate.from_messages(
    [("system", supervisor_prompt),
     MessagesPlaceholder(variable_name="messages"),
     ("system", "Given the conversation above, who should act next?"
                "Or should we FINISH?"
                "Select one of {options}")]
).partial(
    options=str(options),
    members=", ".join(members)
)

class routeResponse(BaseModel):
    next: Literal[*options]

supervisor_chain = sv_prompt | llm.with_structured_output(routeResponse)

def supervisor_agent(state):
    """supervisor agent"""
    result = supervisor_chain.invoke(state)
    # supervisor의 응답을 messages에 추가
    return {"next": result.next, 
            "agent_name": result.next, 
            "messages": []
            }


if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="AI PM 되려면 어케 해야 함?")]}
    result = supervisor_chain.invoke(test_state)
    print(result.next)