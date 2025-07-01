import os
import sys
import locale
import uuid
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sample_data import mentor1, mentee

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

# 멘토 데이터 추출
mentor_data = mentor1
mentee_data = mentee

# json 멘토 데이터 str로 파싱
mentor_safe_str = str(mentor_data).replace('{', '{{').replace('}', '}}')

system_template = f"""
You are a senior expert providing mentoring to {mentee_data}.

[Mentee Information]
{mentee_data}

[Mentor Information]
{mentor_safe_str}

[Role Guidelines]
1. Base advice strictly on mentor data fields (e.g., **group_name**, **current_position**, **common_project**).  
2. Start with "In my experience…" or "In actual projects…".  
3. Mention project names, scale (e.g., "a 50 billion KRW project"), tech stacks (e.g., Spring Boot, AWS).  
4. Balance success stories and failure lessons.  
5. Tailor advice to the mentee's context.  
6. Keep sentences short and break into 1–2 sentence paragraphs.  
7. End each section with a follow-up question to invite dialogue.

[Conversation Style]
- Use polite Korean honorifics (존댓말).  
- Maintain a friendly yet professional tone.  

Always respond based on actual data and examples.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="conversation"),
    ("human", "{user_input}")
])

# 체인 생성
chain = chat_prompt | llm

# 세션 저장소
chat_sessions: Dict[str, Dict] = {}

def create_chat_session() -> str:
    """새로운 채팅 세션 생성"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {"conversation": []}
    return session_id

def chat_with_mentor(user_input: str, session_id: str = None):
    """멘토와 대화"""
    try:
        if not session_id or session_id not in chat_sessions:
            session_id = create_chat_session()
        
        conversation_history = chat_sessions[session_id]["conversation"]
        
        # 체인으로 응답 생성
        response = chain.invoke({
            "conversation": conversation_history,
            "user_input": user_input
        })
        
        # 대화 히스토리에 추가
        conversation_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response.content)
        ])
        
        return {
            "user_id": "",
            "chat_summary": "",
            "answer": response.content,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "user_id": "",
            "chat_summary": "",
            "answer": "",
            "success": False,
            "error": str(e)
        }

def reset_session(session_id: str = None):
    """세션 초기화"""
    return create_chat_session()

def safe_input(prompt):
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print("입력 인코딩 오류가 발생했습니다. 다시 입력해주세요.")
        return safe_input(prompt)

if __name__ == "__main__":
    print(f"=== {mentor_data['group_name']} 멘토와의 대화 시작 ===")
    print(f"멘토: {mentor_data['current_position']} ({mentor_data['experience_years']} 경력)")
    print("명령어: 'quit' (종료), 'reset' (초기화)")
    print("-" * 50)
    
    session_id = create_chat_session()
    
    while True:
        user_input = safe_input("\n멘티: ")
        
        if user_input.lower() in ['quit', '종료', 'exit']:
            print("대화를 종료합니다.")
            break
            
        if user_input.lower() == 'reset':
            session_id = reset_session(session_id)
            print("대화가 초기화되었습니다.")
            continue
        
        result = chat_with_mentor(user_input, session_id)
        
        if result["success"]:
            print(f"멘토: {result['answer']}")
        else:
            print(f"오류: {result['error']}")