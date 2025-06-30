import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import careerTitle_prompt

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

os.environ["OPENAI_API_KEY"] = api_key
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = os.getenv("TEMPERATURE")
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# CareerTitle 프롬프트
cs_prompt = PromptTemplate(
    input_variables=["messages"],
    template=careerTitle_prompt
)

# CareerTitle 체인
careerSummary_chain = cs_prompt | llm | StrOutputParser()

def CareerTitle_invoke(backend_data: dict, config=None) -> dict:
    """CareerTitle Chain 실행 함수"""
    
    result = careerSummary_chain.invoke({
        "messages": backend_data
    })
    
    new_messages = []
    new_messages.append(AIMessage(content=result, name="CareerTitle"))

    user_info = backend_data.get("user_info", {})
    profile_id = str(user_info.get('profileId', ''))
    
    return {
        "profile_id": profile_id,
        "messages": new_messages
    }

