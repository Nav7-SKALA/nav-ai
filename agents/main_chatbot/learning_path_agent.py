# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# from config import AGENT_ROOT, MODEL_NAME, TEMPERATURE
# from prompt import learningPath_prompt

# import sys
# sys.path.append(AGENT_ROOT)

# from tools.search_coursera_courses import search_coursera_courses
# from tools.vector_search_tool import lecture_search
# from tools.tavily_search_tool import search_conferences, search_certifications
# from tools.google_news_tool import google_news_search
# # 환경변수 로드
# load_dotenv()

# # LLM 초기화
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

# os.environ["OPENAI_API_KEY"] = api_key
# llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# # Tools 리스트
# tools = [lecture_search,search_coursera_courses,search_conferences,search_certifications,google_news_search]

# # LLM에 tool 바인딩
# llm_with_tools = llm.bind_tools(tools)

# # LearningPath 프롬프트
# prompt = PromptTemplate.from_template(learningPath_prompt)

# # 간단한 체인
# chain = prompt | llm_with_tools

# def learningPath_invoke(state: dict, config=None) -> dict:
#     """LearningPath Agent 실행 함수"""
#     messages_text = "\n".join([
#         msg.content for msg in state.get("messages", [])
#         if hasattr(msg, 'content')
#     ])
    
#     try:
#         result = chain.invoke({"input": messages_text})
#         response_text = result.content
#         print("tool calling 답변!!!")
#         print("="*10)
#         print(response_text)
        
#         # Tool calls가 있으면 실행
#         if result.tool_calls:
#             tool_results = []
#             for tool_call in result.tool_calls:
#                 tool_name = tool_call["name"]
#                 tool_args = tool_call["args"]
                
#                 # 해당 도구 찾아서 실행
#                 for tool in tools:
#                     if tool.name == tool_name:
#                         tool_result = tool.invoke(tool_args)
#                         tool_results.append(f"{tool_name}: {tool_result}")
#                         break
            
#             # 도구 결과를 포함해서 다시 LLM 호출
#             follow_up_prompt = f"{messages_text}\n\n도구 실행 결과:\n" + "\n".join(tool_results) + "\n\n위 검색 결과를 바탕으로 사용자의 요청에 적절하게 답변해주세요."
#             final_result = llm.invoke([("human", follow_up_prompt)])
#             response_text = final_result.content
#         else:
#             response_text = result.content
            
#     except Exception as e:
#         response_text = f"학습 경로 추천 중 오류가 발생했습니다: {str(e)}"
    
#     new_messages = []#list(state.get("messages", []))
#     new_messages.append(AIMessage(content=response_text, name="LearningPath"))
    
#     return {
#         **state,
#         "messages": new_messages
#     }

# def learningPath_node(state):
#     """LearningPath 노드 함수"""
#     return learningPath_invoke(state)


# if __name__ == "__main__":
#     # 테스트
#     test_state = {"messages": [HumanMessage(content="cloud 분야 컨퍼런스 추천해줘")]}
#     result = learningPath_invoke(test_state)
#     print(result["messages"][-1].content)

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
from tools.vector_search_tool import lecture_search
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
tools = [lecture_search,search_coursera_courses,search_conferences,search_certifications,google_news_search]

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
    verbose=True,
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


if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="AI 개발자가 되고 싶은데 어떻게 공부해야 할까요?")]}
    result = learningPath_invoke(test_state)
    print(result["messages"][-1].content)