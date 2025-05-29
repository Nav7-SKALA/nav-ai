from langchain_core.messages import HumanMessage
from graph import graph

if __name__ == "__main__":
    user_query = "프론트엔드 개발자가 되려면?"
    
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "input_query": user_query,
        "information": ""
    }
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        if hasattr(msg, 'name') and msg.name:
            print(f"\n[{msg.name}]: {msg.content}")
        else:
            print(f"\n[사용자]: {msg.content}")