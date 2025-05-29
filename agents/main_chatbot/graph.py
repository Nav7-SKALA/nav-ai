from langgraph.graph import StateGraph, START, END
from agents import (
    AgentState, 
    members, 
    supervisor_agent, 
    careerSummary_node, 
    learningPath_node, 
    exception_node
)

def create_workflow():
    """LangGraph Workflow 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("CareerSummary", careerSummary_node)
    workflow.add_node("LearningPath", learningPath_node)
    workflow.add_node("EXCEPTION", exception_node)
    workflow.add_node("supervisor", supervisor_agent)
    
    # 멤버에서 supervisor로 가는 엣지
    for member in members:
        workflow.add_edge(member, "supervisor")
    
    # 조건부 맵핑
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"],
        conditional_map
    )
    
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()

# 그래프 생성
graph = create_workflow()