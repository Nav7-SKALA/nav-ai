from career_summary_agent import career_summary_node
from employee_info_node import search_profile_by_id
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# State 정의
class CareerState(TypedDict):
    messages: list
    employee_id: Optional[str]
    profile_id: Optional[str]
    profile_data: list

# LangGraph 생성
def create_career_graph():
    workflow = StateGraph(CareerState)
    
    # 노드 추가
    workflow.add_node("search_profile", search_profile_by_id)
    workflow.add_node("career_summary", career_summary_node)
    
    # 엣지 연결
    workflow.set_entry_point("search_profile")
    workflow.add_edge("search_profile", "career_summary")
    workflow.add_edge("career_summary", END)
    
    return workflow.compile()

# 그래프 실행 함수
def run_career_analysis(employee_id=None, profile_id=None):
    """LangGraph로 커리어 분석 실행"""
    if not employee_id and not profile_id:
        return {"messages": [AIMessage(content="employee_id 또는 profile_id를 입력해주세요.")]}
    
    graph = create_career_graph()
    
    initial_state = {
        "messages": [],
        "employee_id": employee_id,
        "profile_id": profile_id,
        "profile_data": []
    }
    
    try:
        result = graph.invoke(initial_state)
        return result
    except Exception as e:
        return {"messages": [{"content": f"그래프 실행 중 오류: {str(e)}"}]}

if __name__ == "__main__":
    # 테스트 실행
    result = run_career_analysis(employee_id="EMP-525170")
    print(result["messages"][-1].content)