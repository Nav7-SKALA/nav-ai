from typing import TypedDict, Optional, List, Dict, Literal
from typing import Annotated, Sequence
from pydantic import BaseModel
import operator

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

from nav7_test.langgraph import options

class GraphState(TypedDict):
    input: str
    responses: Dict[str, str]
    history: List[str]
    current_agent: Optional[str]
    next_agent: Optional[str]


class routeResponse(BaseModel):
    next: Literal[*options]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str   # supervisor 결과 (다음 노드 지시)