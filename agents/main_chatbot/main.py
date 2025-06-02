import sys
import os

# 현재 파일의 상위 디렉토리 (nav-ai)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

from graph import create_workflow #* #graph

if __name__ == "__main__":
    user_query = "AI쪽 PM이 되고 싶으면 어떻게 해야 하나요?"

    graph = create_workflow()

    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }

    result = graph.invoke(initial_state)

    for msg in result["messages"]:
        print(msg)