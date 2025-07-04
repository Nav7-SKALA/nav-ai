from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Dict

load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = os.getenv("TEMPERATURE")

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

chat_summary_prompt = """
다음은 사용자와 AI 간의 대화 내역입니다:

대화 내역: {chat_history}

위 전체 대화 흐름을 바탕으로, 사용자에게 전달된 주요 내용과 요점을 한국어로 2~3문단으로 요약해 주세요.
- 사용자의 주요 질문이나 관심사 파악
- AI가 제공한 핵심 조언이나 정보 정리
- 대화의 맥락과 진행 방향 유지
- 중복 없이 간결하게 정리
- 사용자가 이해하기 쉽게 작성
- 만약 입력된 대화가 없으면 첫 대화임을 알려주세요.
"""

def chat_summary(chat_histroy: str):
    prompt_template = PromptTemplate(
        input_variables=["chat_history"],
        template=chat_summary_prompt
    )
    
    chain = prompt_template | llm
    result = chain.invoke({"chat_history": chat_histroy})
    
    return result.content