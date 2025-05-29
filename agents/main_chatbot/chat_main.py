from langchain_core.messages import HumanMessage
from graph import graph

def run_career_agent(user_input: str):
    """커리어 에이전트 실행"""
    print(f"사용자 입력: {user_input}\n")
    print("=" * 50)
    
    for step in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        if "__end__" not in step:
            for key, value in step.items():
                print(f"[{key}]")
                if "messages" in value:
                    print(value["messages"][-1].content)
                else:
                    print(value)
                print("-" * 30)

def main():
    """메인 실행 함수"""
    # 예시 질문들
    # test_queries = [
    #     "AI쪽 PM이 되고 싶으면 어떻게 해야 하나요?",
    #     "데이터 사이언티스트가 되고 싶습니다. 어떤 준비를 해야 할까요?",
    #     "3년차 백엔드 개발자인데 DevOps 엔지니어로 전환하고 싶어요."
    # ]
    print("*"*30)
    test_query = "데이터 사이언티스트가 되고 싶습니다. 어떤 준비를 해야 할까요?"
    run_career_agent(test_query)
    
if __name__ == "__main__":
    main()