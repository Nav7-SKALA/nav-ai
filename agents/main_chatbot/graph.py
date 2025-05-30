from typing import TypedDict, Annotated, List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import operator

from langgraph.graph import StateGraph, START, END
from states import GraphState, routeResponse, AgentState

from supervisor_agent import supervisor_agent
from role_model_agent import roleModel_node
from career_summary_agent import careerSummary_node
from learning_path_agent import learningPath_node
from exception_agent import exception_node

# 멤버 및 옵션 정의
members = ["CareerSummary", "LearningPath", "RoleModel", "EXCEPTION"]
options = ["FINISH"] + members


# 기존 State 유지하면서 안전하게 처리
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    input_query: str

def create_safe_workflow():
    """안전한 워크플로우 생성"""
    
    workflow = StateGraph(AgentState)
    
    # 노드들 추가 (기존 코드 유지)
    workflow.add_node("CareerSummary", careerSummary_node)
    workflow.add_node("LearningPath", learningPath_node)
    workflow.add_node("RoleModel", roleModel_node)
    workflow.add_node("EXCEPTION", exception_node)
    workflow.add_node("supervisor", supervisor_agent)

    # 엣지 추가
    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x.get("next", "FINISH"),  # 안전한 접근
        conditional_map
    )
    
    workflow.add_edge(START, "supervisor")
    
    # 체크포인터 없이 컴파일
    graph = workflow.compile()
    return graph

# 안전한 초기 상태 생성
def create_safe_initial_state(user_query: str):
    """타입 안전한 초기 상태"""
    
    return {
        "messages": [HumanMessage(content=user_query)],
        "next": "",
        "input_query": user_query
    }

graph = create_safe_workflow()
