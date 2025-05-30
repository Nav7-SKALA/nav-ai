import functools
import operator
from typing import Annotated, TypedDict, Sequence, Literal
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from prompt import *
from tools import *
from states import GraphState, routeResponse, AgentState
from config import MODEL_NAME
from agents import *


members = ["CareerSummary", "LearningPath", "RoleModel", "EXCEPTION"]
options = ["FINISH"] + members


workflow = StateGraph(AgentState)

workflow.add_node("CareerSummary", careerSummary_node)
workflow.add_node("LearningPath", learningPath_node)
workflow.add_node("RoleModel", roleModel_node)
workflow.add_node("EXCEPTION", exception_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

workflow.add_conditional_edges(
    "supervisor", 
    lambda x: x["next"], # name of next node
    conditional_map)     # path of next node

workflow.add_edge(START, "supervisor")

graph = workflow.compile()

print(conditional_map)


if __name__ == "__main__":

    for s in graph.stream({"messages": [HumanMessage(content="AI쪽 PM이 되고 싶으면 어떻게 해야 하나요?")]}):
        if "__end__" not in s: # state
            print(s)
            print("----")

