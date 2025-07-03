from langgraph.graph import StateGraph, END, START
from agents.main_chatbot.developstate import DevelopState
from typing import Literal, Dict, Any
from agents.main_chatbot.agent import intent_analize, rewrite, exception, path, role_model, trend, chat_summary, ragwrite
from agents.main_chatbot.agent import similar_roadmap, future_career_recommend
from db.mongo import get_session_data
from db.postgres import get_career_summary
from agents.main_chatbot.performance_monitor import monitor

## Router
# 1차 분기: EXCEPTION 여부 판단
def route_from_intent(state: DevelopState) -> Literal["rewriter_node", "EXCEPTION"]:
    intent = state['intent']
    if "EXCEPTION" in intent:
        return "EXCEPTION"
    else:
        return "rewriter_node"

# 2차 분기: 정상 에이전트 분기
def route_to_agent(state: DevelopState) -> Literal["path_recommend", "role_model", "trend_path", "career_goal"]:
    intent = state['intent']
    if "path_recommend" in intent:
        return "path_recommend" 
    elif "role_model" in intent:
        return "role_model"
    elif "trend_path" in intent:
        return "trend_path"
    elif "career_goal" in intent:
        return "career_goal"
    else:
        return "trend_path"
    
#create workflow
def createworkflow():
    workflow = StateGraph(DevelopState)
    workflow.add_node("intent_analize", intent_analize)
    workflow.add_node("rewriter_node", rewrite)
    workflow.add_node("ragwriter_node", ragwrite)
    workflow.add_node("EXCEPTION", exception) 
    workflow.add_node("recommend_roadmap", path)
    ## add
    workflow.add_node("similar_roadmap", similar_roadmap)
    workflow.add_node("role_model", role_model)
    workflow.add_node("future_career", future_career_recommend)
    workflow.add_node("trend_path", trend)
    workflow.add_node("chatting_summary", chat_summary)
    workflow.add_edge(START, "intent_analize")
    workflow.add_conditional_edges(
        "intent_analize",
        route_from_intent,
        {
            "rewriter_node": "rewriter_node",  # 정상 의도 → 재작성
            "EXCEPTION": "EXCEPTION"           # 예외 의도 → 바로 예외 처리
        }
    )
    workflow.add_edge("rewriter_node", "ragwriter_node")
    workflow.add_conditional_edges(
        "ragwriter_node",
        route_to_agent,
        {
            "path_recommend": "similar_roadmap",
            "role_model": "role_model", 
            "trend_path": "trend_path",
            "career_goal": "future_career"
        }
    )
    # add
    workflow.add_edge("similar_roadmap", "recommend_roadmap")
    workflow.add_edge("recommend_roadmap", "chatting_summary")
    workflow.add_edge("role_model", "chatting_summary")
    workflow.add_edge("trend_path", "chatting_summary")
    workflow.add_edge("future_career", "chatting_summary")
    workflow.add_edge("EXCEPTION", "chatting_summary")  
    workflow.add_edge("chatting_summary", END)  
    graph = workflow.compile()
    return graph

def create_initial_state(user_id: str, input_query: str, career_summary: str, chat_summary: str) -> DevelopState:
    return {
        "user_id": user_id,
        "input_query": input_query,
        "career_summary": career_summary,
        "chat_summary":chat_summary,
        "intent": "",
        "rewrited_query": "",
        "similar_career": {},
        "rag_query": "",
        "result": {},  
        "messages": [],
    }

async def run_mainchatbot(user_id: str, input_query: str, session_id: str):
    monitor.start_monitoring()

    graph = createworkflow()
    result = await graph.ainvoke(create_initial_state(user_id, input_query, get_career_summary(user_id),get_session_data(session_id)))
    
    monitor.print_summary()
    
    return result