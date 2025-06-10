import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from config import MODEL_NAME, TEMPERATURE
from employee_info_node import search_profile_by_id
from prompt import persona_chat_prompt

import sys
import locale
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# 전역 변수로 대화 히스토리 관리
conversation_history = []

# 코드 상단에 추가
if sys.platform == "darwin":  # macOS
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

# input 함수 대신 이렇게 사용
def safe_input(prompt):
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print("입력 인코딩 오류가 발생했습니다. 다시 입력해주세요.")
        return safe_input(prompt)

def get_employee_info(employee_id):
    """사원번호로 직원 정보 가져오기"""
    try:
        search_state = {"employee_id": employee_id}
        result = search_profile_by_id(search_state)
        
        if not result.get("profile_data"):
            return None
        
        return result["profile_data"][0]
        
    except Exception as e:
        print(f"오류: {str(e)}")
        return None

def init_chat(employee_id):
    """대화 초기화"""
    global conversation_history
    conversation_history = []
    
    employee_info = get_employee_info(employee_id)
    
    if not employee_info:
        return False
    
    # 역할 설정
    persona_prompt = persona_chat_prompt.format(employee_info=employee_info)
    
    conversation_history.append(SystemMessage(content=persona_prompt))
    return True

def chat(user_message):
    """대화 진행"""
    global conversation_history
    
    if not conversation_history:
        return "먼저 init_chat()으로 대화를 초기화해주세요."
    
    # 사용자 메시지 추가
    conversation_history.append(HumanMessage(content=user_message))
    
    # LLM 응답
    response = llm.invoke(conversation_history)
    
    # AI 응답 히스토리에 추가
    conversation_history.append(AIMessage(content=response.content))
    
    return response.content

def reset_chat():
    """대화 리셋"""
    global conversation_history
    conversation_history = []
    print("대화가 초기화되었습니다.")

if __name__ == "__main__":
    # 대화 시작
    employee_id = "EMP-525170"
    
    if not init_chat(employee_id):
        print("사원 정보를 찾을 수 없습니다.")
        exit()
    
    print("=== 대화 시작 (종료: quit, 초기화: reset) ===")
    
    while True:
        user_input = safe_input("\n사용자: ")
        
        if user_input.lower() in ['quit', '종료', 'exit']:
            print("대화를 종료합니다.")
            break
            
        if user_input.lower() == 'reset':
            reset_chat()
            if not init_chat(employee_id):
                print("사원 정보를 찾을 수 없습니다.")
                break
            continue
        
        response = chat(user_input)
        print(f"사원: {response}")
