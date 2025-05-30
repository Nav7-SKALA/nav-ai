import os
import operator
from typing import Annotated, TypedDict, Sequence, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from states import *
from prompt import supervisor_prompt
from config import MODEL_NAME, TEMPERATURE

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

supervisor_chain = supervisor_prompt | llm.with_structured_output(routeResponse)

def supervisor_agent(state):
    """supervisor agent"""
    return supervisor_chain.invoke(state)
