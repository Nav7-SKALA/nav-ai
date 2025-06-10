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

# if __name__ == "__main__":
#     # 테스트 (employee_info_node 결과를 시뮬레이션)
#     test_state = {
#         "profile_data": [
#             {
#                 "employeeId": "EMP-198261",
#                 "content": '**EMP-198261** 직원은 14년차부터 15년차까지의 경험으로, 금융 도메인에서 사업개발 및 제안PM 역할을 담당했습니다. 주요 업무는 금융기관 1기 ITO 제안 및 협상이며, 프로젝트 규모는 (초대형) 500억입니다. 활용한 기술 스택은 Domain Expert이며, 관련된 직무 영역은 사업관리/개발/제안, PL입니다. \n\n**EMP-198261** 직원은 29년차부터 30년차까지의 경험으로, 금융 도메인에서 수행PM 역할을 담당했습니다. 주요 업무는 금융기관 2기 ITO 수행이며, 프로젝트 규모는 (초대형) 500억입니다. 활용한 기술 스택은 Infra PM -- 대형PM이며, 관련된 직무 영역은 Project Mgmt. 직무 -- 대형PM입니다. \n\n**EMP-198261** 직원은 30년차부터 31년차까지의 경험으로, 금융 도메인에서 수행PM 역할을 담당했습니다. 주요 업무는 금융기관 3기 ITO 제안 및 수행이며, 프로젝트 규모는 (초대형) 500억입니다. 활용한 기술 스택은 Infra PM -- 대형PM이며, 관련된 직무 영역은 Project Mgmt. 직무 -- 대형PM입니다. \n\n**EMP-198261** 직원은 8년차부터 8년차까지의 경험으로, 제조 도메인에서 DBA 역할을 담당했습니다. 주요 업무는 에너지화학기업 ERP 구축이며, 프로젝트 규모는 (대형) 100억 이상 ~ 500억 미만입니다. 활용한 기술 스택은 Middleware/Database Eng이며, 관련된 직무 영역은 Cloud/Infra Eng.직무입니다. \n\n**EMP-198261** 직원은 27년차부터 27년차까지의 경험으로, 유통/서비스 도메인에서 제안PM 및 협상 역할을 담당했습니다. 주요 업무는 유통그룹 2기 ITO 제안 및 협상이며, 프로젝트 규모는 (초대형) 500억입니다. 활용한 기술 스택은 Domain Expert이며, 관련된 직무 영역은 사업관리/개발/제안, PL입니다. \n\n**EMP-198261** 직원은 7년차부터 7년차까지의 경험으로, 금융 도메인에서 DBA 역할을 담당했습니다. 주요 업무는 제1금융권 ER 구축이며, 프로젝트 규모는 (중소형) 20억 이상~50억 미만입니다. 활용한 기술 스택은 Middleware/Database Eng이며, 관련된 직무 영역은 Cloud/Infra Eng.직무입니다. \n\n**EMP-198261** 직원은 29년차부터 29년차까지의 경험으로, 금융 도메인에서 제안PM 역할을 담당했습니다. 주요 업무는 제1금융권 센터 이전 컨설팅 제안이며, 프로젝트 규모는 (소형) 20억 미만입니다. 활용한 기술 스택은 Domain Expert, Infra PM이며, 관련된 직무 영역은 사업관리/개발/제안, PL, Project Mgmt. 직무입니다. \n\n**EMP-198261** 직원은 15년차부터 15년차까지의 경험으로, 금융 도메인에서 사업개발 및 제안PM 역할을 담당했습니다. 주요 업무는 제2금융권 전사콜인프라 개선 제안 및 협상이며, 프로젝트 규모는 (대형) 100억 이상 ~ 500억 미만입니다. 활용한 기술 스택은 Domain Expert이며, 관련된 직무 영역은 사업관리/개발/제안, PL입니다. \n\n**EMP-198261** 직원은 1년차부터 5년차까지의 경험으로, 유통/.서비스 도메인에서 (Mainframe OS/390, DB2) 운영자 역할을 담당했습니다. 주요 업무는 종합상사 시스템 운영이며, 프로젝트 규모는 nan입니다. 활용한 기술 스택은 System/Network Eng이며, 관련된 직무 영역은 Cloud/Infra Eng.직무입니다. \n\n**EMP-198261** 직원은 13년차부터 14년차까지의 경험으로, 제조 도메인에서 사업 개발 및 제안PM 역할을 담당했습니다. 주요 업무는 항공사 ITO 제안 및 협상이며, 프로젝트 규모는 (초대형) 500억입니다. 활용한 기술 스택은 Domain Expert이며, 관련된 직무 영역은 사업관리/개발/제안, PL입니다.'
#             }
#         ]
#     }
#     result = career_summary_node(test_state)
#     print(result["messages"][-1].content)