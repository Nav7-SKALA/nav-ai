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

MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = os.getenv("TEMPERATURE")
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# CareerTitle 프롬프트
ct_prompt = PromptTemplate(
    input_variables=["messages"],
    template=careerTitle_prompt
)

# CareerTitle 체인
careerTitle_chain = ct_prompt | llm | StrOutputParser()

def CareerTitle_invoke(backend_data: dict) -> dict:
    """CareerTitle Chain 실행 함수"""
    try:
        result = careerTitle_chain.invoke({
            "messages": backend_data
        })
        
        user_info = backend_data.get("user_info", {})
        profile_id = user_info.get('profileId', '')
        
        return {
            "profile_id": profile_id,
            "career_title": result
        }
    except Exception as e:
        return {
            "profile_id": "",
            "career_title": f"오류 발생: {str(e)}"
        }

