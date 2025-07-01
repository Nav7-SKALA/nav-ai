import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import sys
import locale

load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = os.getenv("TEMPERATURE")

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# 세션별 대화 히스토리 저장 (메모리 기반)
chat_sessions: Dict[str, Dict] = {}

# 프롬프트 템플릿
persona_chat_prompt = """You are employee {employee_id}, a senior expert with many years of rich experience.

[Your Career Information]
{employee_content}

[Role Guidelines]
- Share actual experiences based on the above career
- Mention specific project names, technologies, and domain experiences
- Act as a mentor advising junior developers/PMs
- Provide practical tips from real work experience
- Maintain a friendly yet professional tone

[Conversation Style]
- Use expressions like "In my experience..." "In actual projects..."
- Mention specific numbers or scales (e.g., "In a 50 billion KRW project...")
- Share both failure stories and success stories in a balanced way
- Provide personalized advice suited to the questioner's situation

Now please naturally converse with users and share your valuable experiences."""

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

def create_chat_session(employee_id: str) -> str:
    """새로운 채팅 세션 생성"""
    session_id = str(uuid.uuid4())
    
    employee_info = get_employee_info(employee_id)
    if not employee_info:
        return None
    
    # 프롬프트에서 직접 employee_info 값들을 사용
    persona_prompt = persona_chat_prompt.format(
        employee_id=employee_info.get('employeeId', employee_id),
        employee_content=employee_info.get('content', '정보를 찾을 수 없습니다.')
    )
    
    chat_sessions[session_id] = {
        "employee_id": employee_id,
        "employee_info": employee_info,
        "conversation_history": [SystemMessage(content=persona_prompt)],
        "created_at": datetime.now(),
        "last_used": datetime.now()
    }
    
    return session_id

def chat_with_session(employee_id: str, user_message: str, session_id: str = None):
    """세션 기반 채팅"""
    try:
        # 세션 ID가 없거나 존재하지 않으면 새로 생성
        if not session_id or session_id not in chat_sessions:
            session_id = create_chat_session(employee_id)
            if not session_id:
                return {"success": False, "error": "사원 정보를 찾을 수 없습니다."}
        else:
            # 기존 세션이 다른 employee_id를 사용하는 경우 새로 생성
            if chat_sessions[session_id]["employee_id"] != employee_id:
                session_id = create_chat_session(employee_id)
                if not session_id:
                    return {"success": False, "error": "사원 정보를 찾을 수 없습니다."}
        
        session_data = chat_sessions[session_id]
        
        # 사용자 메시지 추가
        session_data["conversation_history"].append(
            HumanMessage(content=user_message)
        )
        
        # LLM 응답
        response = llm.invoke(session_data["conversation_history"])
        
        # AI 응답 히스토리에 추가
        session_data["conversation_history"].append(
            AIMessage(content=response.content)
        )
        
        # 마지막 사용 시간 업데이트
        session_data["last_used"] = datetime.now()
        
        return {
            "success": True,
            "response": response.content,
            "session_id": session_id
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def init_chat(employee_id):
    """대화 초기화 (기존 코드와의 호환성을 위해 유지)"""
    session_id = create_chat_session(employee_id)
    return session_id is not None

def chat(user_message, session_id=None):
    """대화 진행 (기존 코드와의 호환성을 위해 유지 - 하지만 세션 ID 필요)"""
    if not session_id or session_id not in chat_sessions:
        return "먼저 init_chat()으로 대화를 초기화해주세요."
    
    session_data = chat_sessions[session_id]
    
    # 사용자 메시지 추가
    session_data["conversation_history"].append(HumanMessage(content=user_message))
    
    # LLM 응답
    response = llm.invoke(session_data["conversation_history"])
    
    # AI 응답 히스토리에 추가
    session_data["conversation_history"].append(AIMessage(content=response.content))
    
    return response.content

def reset_chat():
    """대화 리셋"""
    global chat_sessions
    chat_sessions.clear()
    print("모든 대화 세션이 초기화되었습니다.")

def reset_session(session_id: str):
    """특정 세션 리셋"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return True
    return False

if __name__ == "__main__":
    # 대화 시작
    employee_id = "EMP-525170"
    
    session_id = create_chat_session(employee_id)
    if not session_id:
        print("사원 정보를 찾을 수 없습니다.")
        exit()
    
    print("=== 대화 시작 (종료: quit, 초기화: reset) ===")
    
    while True:
        user_input = safe_input("\n사용자: ")
        
        if user_input.lower() in ['quit', '종료', 'exit']:
            print("대화를 종료합니다.")
            break
            
        if user_input.lower() == 'reset':
            reset_session(session_id)
            session_id = create_chat_session(employee_id)
            if not session_id:
                print("사원 정보를 찾을 수 없습니다.")
                break
            print("대화가 초기화되었습니다.")
            continue
        
        result = chat_with_session(employee_id, user_input, session_id)
        if result["success"]:
            print(f"사원: {result['response']}")
        else:
            print(f"오류: {result['error']}")