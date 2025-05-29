import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import search_coursera_courses, search_conferences, search_certifications, google_news_search

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

# Tools 리스트
tools = [search_coursera_courses, search_conferences, search_certifications, google_news_search]

# LLM에 tool 바인딩
llm_with_tools = llm.bind_tools(tools)

# LearningPath 프롬프트
prompt = PromptTemplate.from_template("""
You are a highly experienced consultant across multiple industries.

Based on the user's career history and past experiences, your task is to:
- Recommend clear professional goals for the user.
- Suggest actionable steps to achieve those goals, including relevant skills to learn, projects to undertake, and certifications to pursue.

Do not repeat information the user already has experience in.  
Focus instead on new, meaningful paths for professional growth tailored to the user’s current situation.

You may use predefined tools to retrieve any additional information needed.  
Organize your final recommendations into categories (e.g., Skills, Projects, Certifications, etc.).

⚠️ All responses must be written in Korean.
""")

# 간단한 체인
chain = prompt | llm_with_tools

def learningPath_invoke(state: dict, config=None) -> dict:
    """LearningPath Agent 실행 함수"""
    messages_text = "\n".join([
        msg.content for msg in state.get("messages", [])
        if hasattr(msg, 'content')
    ])
    
    try:
        result = chain.invoke({"input": messages_text})
        
        # Tool calls가 있으면 실행
        if result.tool_calls:
            tool_results = []
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # 해당 도구 찾아서 실행
                for tool in tools:
                    if tool.name == tool_name:
                        tool_result = tool.invoke(tool_args)
                        tool_results.append(f"{tool_name}: {tool_result}")
                        break
            
            # 도구 결과를 포함해서 다시 LLM 호출
            follow_up_prompt = f"{messages_text}\n\n도구 실행 결과:\n" + "\n".join(tool_results) + "\n\n위 정보를 바탕으로 한국어로 종합적인 학습 경로를 추천해주세요."
            final_result = llm.invoke([("human", follow_up_prompt)])
            response_text = final_result.content
        else:
            response_text = result.content
            
    except Exception as e:
        response_text = f"학습 경로 추천 중 오류가 발생했습니다: {str(e)}"
    
    new_messages = list(state.get("messages", []))
    new_messages.append(HumanMessage(content=response_text, name="LearningPath"))
    
    return {
        **state,
        "messages": new_messages
    }

def learningPath_node(state):
    """LearningPath 노드 함수"""
    return learningPath_invoke(state)