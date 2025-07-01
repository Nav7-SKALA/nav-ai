from typing import TypedDict, Optional, List, Dict, Literal, Any
from typing import Annotated, Sequence
from pydantic import BaseModel
import operator
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

# ✅ 수정된 State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Sequence → List로 변경
    next: str
    input_query: str

def merge_dicts(left: Dict, right: Dict) -> Dict:
    """두 딕셔너리를 병합하는 함수"""
    return {**left, **right}

class DevelopState(TypedDict):
    user_id: str
    chat_summary: str
    input_query: str
    career_summary: str
    intent: str
    rewrited_query: str
    rag_query: str
    result: Annotated[Dict[str, Any], merge_dicts] #Dict[str, Any]
    messages: Annotated[List, add_messages]