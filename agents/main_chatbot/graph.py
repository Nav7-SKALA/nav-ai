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

# ✅ 수정된 State 정의 (List 사용, 중복 제거)
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add] = [] # Sequence → List
    #next: str
    #input_query: str

def supervisor_router(state: AgentState) -> str:
    """supervisor의 결과를 안전하게 라우팅"""
    try:
        # supervisor_agent가 반환하는 값 확인
        next_agent = state.get("next", "")
        
        # 유효한 옵션인지 확인
        if next_agent in options:
            return next_agent
        elif next_agent in members:
            return next_agent
        else:
            print(f"⚠️ 알 수 없는 다음 에이전트: {next_agent}, EXCEPTION으로 라우팅")
            return "EXCEPTION"
            
    except Exception as e:
        print(f"❌ 라우팅 에러: {e}")
        return "EXCEPTION"

def create_workflow():
    """워크플로우 생성"""
    
    try:
        # StateGraph 생성
        workflow = StateGraph(AgentState)
        
        # ✅ 모든 노드 추가
        workflow.add_node("supervisor", supervisor_agent)
        workflow.add_node("CareerSummary", careerSummary_node)
        workflow.add_node("LearningPath", learningPath_node)
        workflow.add_node("RoleModel", roleModel_node)
        workflow.add_node("EXCEPTION", exception_node)
                
        workflow.add_edge(START, "supervisor")
        
        for member in members:
            workflow.add_edge(member, "supervisor")
        
        conditional_map = {}
        for member in members:
            conditional_map[member] = member
        conditional_map["FINISH"] = END
        
        workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,  # 안전한 라우터 함수 사용
            conditional_map
        )

        graph = workflow.compile()
        
        return graph
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

def create_initial_state(user_query: str) -> AgentState:
    """안전한 초기 상태 생성"""
    
    return {
        "messages": [HumanMessage(content=user_query)],
        #"next": "",
        #"input_query": user_query
    }

def run_workflow(user_query: str):
    """워크플로우 실행"""
    
    try:
        print("=== 워크플로우 실행 시작 ===")
        
        # 1. 워크플로우 생성
        print("1. 워크플로우 생성 중...")
        graph = create_workflow()
        
        # 2. 초기 상태 생성
        print("2. 초기 상태 생성 중...")
        initial_state = create_initial_state(user_query)
        print(f"초기 상태: {initial_state}")
        
        # 3. 워크플로우 실행
        print("3. 워크플로우 실행 중...")
        result = graph.invoke(initial_state)
        print(f"최종 결과: {result}")
        print("✅ 워크플로우 실행 성공")

        return result
        
    except Exception as e:
        print(f"❌ 워크플로우 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# 스트리밍 실행 함수 (디버깅용)
def run_with_streaming(user_query: str, graph):
    """스트리밍으로 실행 과정 확인"""
    
    if graph is None:
        print("❌ 그래프가 생성되지 않았습니다")
        return None
    
    try:
        initial_state = create_initial_state(user_query)
        
        print(f"사용자 질문: {user_query}")
        print("=" * 50)
        
        step = 0
        for chunk in graph.stream(initial_state, stream_mode="values"):
            step += 1
            print(f"\n--- Step {step} ---")
            
            if "next" in chunk:
                print(f"다음 에이전트: {chunk['next']}")
            
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"메시지: {last_message.content}")
        
        print("\n✅ 스트리밍 실행 완료")
        
    except Exception as e:
        print(f"❌ 스트리밍 실행 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    # ✅ 그래프 생성 (모듈 로드 시)
    try:
        graph = create_workflow()
        print("✅ 그래프 생성 완료")
    except Exception as e:
        print(f"❌ 그래프 생성 실패: {e}")
        graph = None

    test_queries = [
        #"커리어 상담을 받고 싶습니다",
        "AI 개발자가 되고 싶은데 어떻게 공부해야 할까요?",
        #"성공한 개발자들의 사례를 알고 싶어요",
        #"오늘 날씨가 어때요?"  # 예외 처리 테스트
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"테스트 질문: {query}")
        print('='*60)
        
        # result = run_workflow(query)
        
        # if result:
        #     print("테스트 성공")
        # else:
        #     print("테스트 실패")

        run_with_streaming(query, graph)