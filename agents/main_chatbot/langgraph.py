from langgraph.graph import StateGraph, END, START
from states import DevelopState
from typing import Literal
from agent import intent_analize, rewrite, exception, path, role_model,trend
## Router
# 1차 분기: EXCEPTION 여부 판단
def route_from_intent(state: DevelopState) -> Literal["rewriter_node", "EXCEPTION"]:
    intent = state['intent']
    if "EXCEPTION" in intent:
        return "EXCEPTION"
    else:
        return "rewriter_node"

# 2차 분기: 정상 에이전트 분기
def route_to_agent(state: DevelopState) -> Literal["path_recommend", "role_model", "trend_path"]:
    intent = state['intent']
    if "path_recommend" in intent:
        return "path_recommend" 
    elif "role_model" in intent:
        return "role_model"
    elif "trend_path" in intent:
        return "trend_path"
    else:
        return "trend_path"
    
#create workflow
def createworkflow():
    workflow = StateGraph(DevelopState)
    workflow.add_node("intent_analize", intent_analize)
    workflow.add_node("rewriter_node", rewrite)
    workflow.add_node("EXCEPTION", exception) 
    workflow.add_node("path_recommend", path)
    workflow.add_node("role_model", role_model)
    workflow.add_node("trend_path", trend)
    workflow.add_edge(START, "intent_analize")
    workflow.add_conditional_edges(
        "intent_analize",
        route_from_intent,
        {
            "rewriter_node": "rewriter_node",  # 정상 의도 → 재작성
            "EXCEPTION": "EXCEPTION"           # 예외 의도 → 바로 예외 처리
        }
    )
    workflow.add_conditional_edges(
        "rewriter_node",
        route_to_agent,
        {
            "path_recommend": "path_recommend", 
            "role_model": "role_model", 
            "trend_path": "trend_path"
        }
    )
    workflow.add_edge("path_recommend", END)
    workflow.add_edge("role_model", END)
    workflow.add_edge("trend_path", END)
    workflow.add_edge("EXCEPTION", END)  
    graph = workflow.compile()
    return graph

def create_initial_state(user_id: str, input_query: str, career_summary: str) -> DevelopState:
    
    return {
        "user_id": user_id,
        "input_query": input_query,
        "career_summary": career_summary,
        "rewrited_query": "",
        "rag_query": "",
        "intent": "",
        "result": {},
        "reference_employees": [],  
        "messages": [],
    }

def run_mainchatbot(user_id: str, input_query: str, career_summary: str):
    graph = createworkflow
    result = graph.invoke(create_initial_state(user_id, input_query, career_summary))
    return result