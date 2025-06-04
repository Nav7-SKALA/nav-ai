import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from prompt import learningPath_prompt


from config import AGENT_ROOT, MODEL_NAME, TEMPERATURE

import sys
sys.path.append(AGENT_ROOT)

from tools.search_coursera_courses import search_coursera_courses
# from tools.vector_search_tool import lecture_search
from tools.tavily_search_tool import search_conferences, search_certifications
from tools.google_news_tool import google_news_search
# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# Tools 리스트
# tools = [lecture_search,search_coursera_courses,search_conferences,search_certifications,google_news_search]
tools = [search_coursera_courses,search_conferences,search_certifications,google_news_search]

# Agent 프롬프트 생성
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        learningPath_prompt
    ),
    (
        "user", 
        "{input}"
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent 정의 
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성 
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)

def learningPath_invoke(state: dict, config=None) -> dict:
    """LearningPath Agent 실행 함수"""
    messages = state.get("messages", [])
    
    # 최신 사용자 메시지 추출
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        return {
            **state,
            "messages": messages + [AIMessage(content="사용자 메시지를 찾을 수 없습니다.", name="LearningPath")]
        }
    
    try:
        # AgentExecutor 실행
        result = agent_executor.invoke({"input": user_message})
        response_text = result["output"]
            
    except Exception as e:
        response_text = f"학습 경로 추천 중 오류가 발생했습니다: {str(e)}"
    
    new_messages = list(messages)
    new_messages.append(AIMessage(content=response_text, name="LearningPath"))
    
    return {
        **state,
        "messages": new_messages
    }

def learningPath_node(state):
    """LearningPath 노드 함수"""
    return learningPath_invoke(state)


# if __name__ == "__main__":
#     # 테스트
#     test_state = {"messages": [HumanMessage(content="AI 개발자가 되고 싶은데 어떻게 공부해야 할까요?")]}
#     result = learningPath_invoke(test_state)
#     print(result["messages"][-1].content)