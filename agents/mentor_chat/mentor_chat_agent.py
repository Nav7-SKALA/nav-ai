import os
import sys
import locale
import uuid
from datetime import datetime
from typing import Dict
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# config 사용해서 경로 설정
from config import BASE_DIR, DB_DIR
sys.path.append(BASE_DIR)
sys.path.append(DB_DIR)

from db.mongo import get_rolemodel_data, get_latest_chat_summary
from mentor_chat_summary import chat_summary
from db.postgres import get_company_direction, get_career_summary

load_dotenv()

# 인코딩 설정
if sys.platform == "darwin":
    try:
        locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    except:
        pass

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = os.getenv("TEMPERATURE")

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# 전역 변수
chat_sessions = []
current_summary = ""
MESSAGE_LIMIT = 10

def safe_input(prompt):
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print("입력 인코딩 오류가 발생했습니다. 다시 입력해주세요.")
        return safe_input(prompt)

def auto_summarize():
    """10개마다 자동 요약"""
    global chat_sessions, current_summary
    
    if len(chat_sessions) >= MESSAGE_LIMIT:
        new_summary = chat_summary(chat_sessions)
        current_summary = new_summary if not current_summary else f"{current_summary} + {new_summary}"
        chat_sessions = chat_sessions[-2:]  # 최근 2개만 유지

def chat_with_mentor(user_id: str, input_query: str, session_id: str, rolemodel_id: str) -> Dict:
    """멘토와 대화하는 메인 함수"""
    global chat_sessions, current_summary
    
    try:
        # 종료 체크
        if input_query.strip().lower() in ['종료', '끝', 'exit', 'quit']:
            final_summary = current_summary
            if chat_sessions:
                final_summary += f" + {chat_summary(chat_sessions)}"
            
            return {
                "user_id": user_id,
                "chat_summary": final_summary,
                "answer": "대화가 종료되었습니다.",
                "success": True,
                "error": None
            }
        
        # 롤모델 데이터 가져오기
        mentor_info = get_rolemodel_data(rolemodel_id)
        
        mentor_data = json.loads(mentor_info["info"])
        mentor_safe_str = str(mentor_data).replace('{', '{{').replace('}', '}}')
        
        # 회사 방향성 가져오기
        direction_data = get_company_direction()
        
        # 멘티 데이터 가져오기
        mentee_info = get_career_summary(user_id)
        
        # 이전 대화 기록 데이터 가져오기
        conversation_history = get_latest_chat_summary(session_id)

        # 프롬프트 설정
        mentor_chat_prompt = """
당신은 {mentee_data}님에게 멘토링을 제공하는 시니어 전문가입니다.

[사용자 요청]
{user_input}

[멘티 정보]
{mentee_data}

[멘토 정보]
{mentor_data}

[회사 미래 방향성]
회사의 미래 방향성: {direction}

[회사 방향성 반영 방법]
제공된 회사의 미래 방향성과 핵심 기술 전략을 분석하여, 사용자의 현재 역량이 회사가 추진하는 기술 영역에서 어떻게 발전할 수 있는지를 고려한 미래 직무를 우선적으로 제시하고, 회사의 장기 비전에 부합하는 전문가 역할을 설계합니다.

[역할 가이드라인]
0. 회사 미래 방향성과 반영 방법을 반드시 준수할 것.  
1. 멘토 데이터 필드(예: **group_name**, **current_position**, **common_project**)를 엄격히 기반으로 조언할 것.  
2. "제 경험에 따르면…", "실제 프로젝트에서는…"으로 시작할 것.  
3. 프로젝트명, 규모(예: "50억 원 규모"), 기술 스택(예: Spring Boot, AWS)을 언급할 것.  
4. 성공 사례와 실패 교훈을 균형 있게 포함할 것.  
5. 멘티 상황 및 사용자 요청({user_input})에 맞춘 조언을 제공할 것.  
6. 문장을 짧게 유지하고 1~2문장 단락으로 구분할 것.  
7. 각 섹션 끝에 대화를 유도하는 후속 질문을 추가할 것.  

[대화 스타일]
- 존댓말을 사용할 것.  
- 친근하면서도 전문적인 톤을 유지할 것.  

항상 실제 데이터와 사례를 기반으로 응답할 것.
"""

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", mentor_chat_prompt),
            ("human", "{user_input}")
        ])

        # 체인 생성
        chain = chat_prompt | llm
        
        # AI 응답 생성
        response = chain.invoke({
            "conversation": conversation_history,
            "user_input": input_query,
            "mentee_data": mentee_info,
            "mentor_data": mentor_safe_str,
            "direction": direction_data
        })
        
        # 대화 히스토리에 추가
        chat_sessions.extend([
            HumanMessage(content=input_query),
            AIMessage(content=response.content)
        ])
        
        # 자동 요약 체크
        auto_summarize()
        
        return {
            "user_id": user_id,
            "chat_summary": "",
            "answer": response.content,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "user_id": user_id,
            "chat_summary": "",
            "answer": "",
            "success": False,
            "error": str(e)
        }

# if __name__ == "__main__":
#     # 테스트용 데이터
#     mentee_info = "1"
#     test_rolemodel_id = "6863baadfefc0f239caad583"
    
#     # 롤모델 정보 가져오기
#     mentor_info = get_rolemodel_data(test_rolemodel_id)
#     mentor_data = json.loads(mentor_info["info"])
    
#     print(f"=== {mentor_data['group_name']} 멘토와의 대화 시작 ===")
#     print(f"멘토: {mentor_data} ({mentor_data['experience_years']} 경력)")
#     print("명령어: 'quit' (종료)")
#     print("-" * 50)
    
#     while True:
#         user_input = safe_input("\n멘티: ")
        
#         result = chat_with_mentor(mentee_info, user_input, "session_123", test_rolemodel_id)
        
#         if result["success"]:
#             print(f"멘토: {result['answer']}")
#             if result["chat_summary"]:  # 종료시에만 요약 출력
#                 print(f"📝 요약: {result['chat_summary']}")
#                 break
#         else:
#             print(f"오류: {result['error']}")