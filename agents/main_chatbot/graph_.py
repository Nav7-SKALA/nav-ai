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

def create_workflow():
    """LangGraph Workflow 생성"""
        
    workflow = StateGraph(AgentState)

    workflow.add_node("CareerSummary", careerSummary_node)
    workflow.add_node("LearningPath", learningPath_node)
    workflow.add_node("RoleModel", roleModel_node)
    workflow.add_node("EXCEPTION", exception_node)
    workflow.add_node("supervisor", supervisor_agent)

    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"], # name of next node
        conditional_map)     # path of next node
    workflow.add_edge(START, "supervisor")
    
    # print(conditional_map)
    graph = workflow.compile()
    return graph

graph = create_workflow()