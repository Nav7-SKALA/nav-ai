from langgraph.graph import StateGraph, START, END

from supervisor_agent import AgentState, members, supervisor_agent
from career_summary_agent import careerSummary_node
from learning_path_agent import learningPath_node
from exception_agent import exception_node

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

if __name__ == "__main__":
    # 테스트용 코드
    from langchain_core.messages import HumanMessage
    
    test_state = {
        "messages": [HumanMessage(content="백엔드 개발자에서 데이터 사이언티스트로 전환하고 싶어요")],
        "next": ""
    }
    
    result = graph.invoke(test_state)
    print("최종 결과:")
    for msg in result["messages"]:
        if hasattr(msg, 'name') and msg.name:
            print(f"[{msg.name}]: {msg.content}")
        else:
            print(f"[User]: {msg.content}")