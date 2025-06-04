import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

from config import MODEL_NAME, TEMPERATURE

from prompt import learning_path_prompt

from tools import search_coursera_courses, search_conferences, search_certifications, google_news_search

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")


llm = ChatOpenAI(model_name=MODEL_NAME, api_key=api_key, temperature=TEMPERATURE)

# Tools 리스트
tools = [search_coursera_courses, search_conferences, search_certifications, google_news_search]

# LLM에 tool 바인딩
llm_with_tools = llm.bind_tools(tools)

# LearningPath 프롬프트
prompt = PromptTemplate.from_template(learning_path_prompt)

# 간단한 체인
learning_path_chain = prompt | llm_with_tools


def learning_path_invoke(state: dict, config=None) -> dict:
    """LearningPath Agent 실행 함수"""
    messages_text = "\n".join([
        msg.content for msg in state.get("messages", [])
        if hasattr(msg, 'content')
    ])
    
    try:
        result = learning_path_chain.invoke({"input": messages_text})
        response_text = result.content
        
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
            follow_up_prompt = f"{messages_text}\n\n도구 실행 결과:\n" + "\n".join(tool_results) + "\n\n위 검색 결과를 바탕으로 사용자의 요청에 적절하게 답변해주세요."
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

def learning_path_node(state):
    """LearningPath 노드 함수"""
    return learning_path_invoke(state)

if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="클라우드 전문가가 되려면 어떻게 학습해야돼?")]}
    result = learning_path_invoke(test_state)
    print(result["messages"][-1].content)

