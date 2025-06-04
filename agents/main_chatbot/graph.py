from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import operator

# 필요한 모듈들 임포트
from supervisor_agent import supervisor_agent
from role_model_agent import roleModel_node
from career_summary_agent import careerSummary_node
from learning_path_agent import learningPath_node
from exception_agent import exception_node


# 멤버 및 옵션 정의
members = ["CareerSummary", "LearningPath", "RoleModel", "EXCEPTION"]
options = ["FINISH"] + members


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add] = [] # Sequence → List
    next: str
    #input_query: str


def supervisor_router(state: AgentState) -> str:
    
    next_agent = state.get("next", "")
        
    # 유효한 옵션인지 확인
    if next_agent in options:
        return next_agent
    elif next_agent in members:
        return next_agent
    else:
        print(f"⚠️ 알 수 없는 다음 에이전트: {next_agent}, EXCEPTION으로 라우팅")
        return "EXCEPTION"


def create_workflow():
    """워크플로우 생성"""
    
    workflow = StateGraph(AgentState)
        
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("CareerSummary", careerSummary_node)
    workflow.add_node("LearningPath", learningPath_node)
    workflow.add_node("RoleModel", roleModel_node)
    workflow.add_node("EXCEPTION", exception_node)
                
    workflow.add_edge(START, "supervisor")
        
    for member in members:
        workflow.add_edge(member, END)
        
    conditional_map = {}
    for member in members:
        conditional_map[member] = member
    conditional_map["FINISH"] = END
        
    workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,  
            conditional_map
    )

    graph = workflow.compile()
        
    return graph


def create_initial_state(user_query: str) -> AgentState:
    
    return {
        "messages": [HumanMessage(content=user_query)],
        "next": "",
        "input_query": user_query
    }


def run_workflow(user_query: str):
    """워크플로우 실행"""
    
    graph = create_workflow()
        
    # 초기 상태 생성
    initial_state = create_initial_state(user_query)
    
    # result = graph.invoke(initial_state)
    # print(result)
    
    result_content = {'content': {}}
    try:
        result = graph.invoke(initial_state)

        content_dict = {"success": True, "result": {}}
        contents = {}
        for message in result['messages'][1:]:
            contents['agent'] = message.name
            if message.name == 'RoleModel':
                import json
                
                rolemodels = json.loads(message.content)
                contents['text'] = rolemodels
            else:
                contents['text'] = message.content
        
        content_dict['result'] = contents
        result_content['content'] = content_dict
        
    # except OpenAIError as e:
    #     result_content['content'] = {
    #         'success': False,
    #         'result': {"OpenAI API error"}
    #     }
    # except RateLimitError as e:
    #     result_content['content'] = {
    #         'success': False,
    #         'result': {"OpenAI rate limit error"}
    #     }
    except Exception as e:
        result_content['content'] = {
            'success': False,
            'result': {f"에러 발생: {e.__class__.__name__}: {e}"}
        }

    return result_content

def run_main_chatbot(user_query: str):
    """메인 챗봇 실행"""

    graph = create_workflow()
    result = run_workflow(user_query)
    return result

if __name__ == "__main__":

    user_query = "롤모델 추천해줘"
    
    result = run_main_chatbot(user_query)
    print(result)

