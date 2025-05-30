import sys
import os

# 현재 파일의 상위 디렉토리 (nav-ai)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

from graph import * #graph

if __name__ == "__main__":
    user_query = "AI쪽 PM이 되고 싶으면 어떻게 해야 하나요?"

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "input_query": user_query,
        "next": "",
        #"information": ""
    }

    result = graph.invoke(initial_state)

    for msg in result["messages"]:
        if hasattr(msg, "name") and msg.name:
            print(f"\n[{msg.name}]: {msg.content}")
        else:
            print(f"\n[사용자]: {msg.content}")

    # for s in graph.stream({"messages": [HumanMessage(content=user_query)]}):
    #     if "__end__" not in s: # state
    #         print(s)
    #         print("----")


# # 실행
# def run_workflow(user_query: str):
#     try:
#         print("워크플로우 생성 중...")
#         graph = create_safe_workflow()
#         print("✅ 워크플로우 생성 성공")
        
#         print("초기 상태 생성 중...")
#         initial_state = create_safe_initial_state(user_query)
#         print(f"초기 상태: {initial_state}")
        
#         print("그래프 실행 시작...")
#         result = graph.invoke(initial_state)
#         print("✅ 실행 성공")
        
#         return result
        
#     except Exception as e:
#         print(f"❌ 에러 발생: {e}")
        
#         # 상세 에러 정보
#         import traceback
#         print("상세 에러:")
#         traceback.print_exc()
        
#         return None

# # 사용
# if __name__ == "__main__":
#     result = run_workflow("커리어 상담을 받고 싶습니다")
#     if result:
#         print("최종 결과:", result)