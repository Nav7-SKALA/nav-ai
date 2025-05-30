import functools

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from prompt import *
from tools import *
from states import GraphState, routeResponse, AgentState
from config import MODEL_NAME


llm = ChatOpenAI(model=MODEL_NAME)


## Supervisor
def supervisor_agent(state):
    supervisor_chain = (
        supervisor_prompt | llm.with_structured_output(routeResponse)
    )
    return supervisor_chain.invoke(state)


## Agents
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=result["messages"][-1].content, name=name)]}

careerSummary_agent = create_react_agent(
    llm,
    tools = [RDB_search, vectorDB_search],
    state_modifier=careerSummary_prompt
)
careerSummary_node = functools.partial(
    agent_node,
    agent=careerSummary_agent,
    name="CareerSummary"
)

learningPath_agent = create_react_agent(
    llm,
    tools=[RDB_search, tavily_tool, google_news_search],
    state_modifier=learningPath_prompt
)
learningPath_node = functools.partial(
    agent_node,
    agent=learningPath_agent,
    name="LearningPath"
)

roleModel_agent = create_react_agent(
    llm,
    tools=[vectorDB_search],
    state_modifier=roleModel_prompt
)
roleModel_node = functools.partial(
    agent_node,
    agent=roleModel_agent,
    name="RoleModel"
)

exception_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=exception_prompt
)
exception_node = functools.partial(
    agent_node,
    agent=exception_agent,
    name="EXCEPTION"
)

