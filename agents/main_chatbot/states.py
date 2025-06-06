from typing import TypedDict, Optional, List, Dict, Literal
from typing import Annotated, Sequence
from pydantic import BaseModel
import operator

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

members = ["CareerSummary", "LearningPath", "RoleModel", "EXCEPTION"]
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[*options]

# ✅ 수정된 State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Sequence → List로 변경
    next: str
    input_query: str