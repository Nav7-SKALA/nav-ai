import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import careerSummary_prompt
from config import  MODEL_NAME, TEMPERATURE
from employee_info_node import search_profile_by_id

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# CareerSummary 프롬프트
cs_prompt = PromptTemplate(
    input_variables=["profile_data"],
    template=careerSummary_prompt
)

# CareerSummary 체인
careerSummary_chain = cs_prompt | llm | StrOutputParser()

def career_summary_node(state):
    """커리어 요약 노드 - profile_data를 받아서 요약"""
    try:
        # employee_info_node에서 가져온 profile_data 추출
        profile_data = state.get("profile_data", [])
        
        if not profile_data:
            return {"messages": [AIMessage(content="요약할 커리어 정보가 없습니다.")]}
        
        # profile_data를 텍스트로 변환
        career_text = ""
        for data in profile_data:
            career_text += f"사원번호: {data.get('employeeId')}\n"
            career_text += f"내용: {data.get('content')}\n\n"
        
        # 커리어 요약 실행
        result = careerSummary_chain.invoke({"profile_data": career_text})
        
        return {
        **state,
        "messages": state.get("messages", []) + [AIMessage(content=result)]
        }
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"커리어 요약 중 오류: {str(e)}")]}

def get_career_summary(employee_id):
   """사원번호로 커리어 요약하기"""
   # 1. 사원 정보 검색
   search_state = {"employee_id": employee_id}
   search_result = search_profile_by_id(search_state)
   
   # 2. 커리어 요약
   summary_result = career_summary_node(search_result)
   return summary_result["messages"][-1].content

if __name__ == "__main__":
   # 테스트
   employee_id = "EMP-198261"
   summary = get_career_summary(employee_id)
   print(summary)