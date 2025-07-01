import os,re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from prompt import careerSummary_prompt

# from tools.vector_search_tool import vectorDB_search
# from tools.postgres_tool import RDB_search

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


# CareerSummary 프롬프트
cs_prompt = PromptTemplate(
    input_variables=["messages"],
    template=careerSummary_prompt
)

# CareerSummary 체인
careerSummary_chain = cs_prompt | llm | StrOutputParser()

def format_career_summary(text: str) -> str:
    """마크다운 형태의 줄바꿈을 HTML 형태로 변환"""
    # \n\n을 <br><br>로 변환 (문단 구분)
    text = text.replace('\n\n', '')
    
    # 단일 \n을 <br>로 변환 (줄바꿈)
    text = text.replace('\n', '')
    
    # 연속된 공백 정리 (선택사항)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def careerSummary_invoke(backend_data: dict) -> dict:
    """CareerSummary Chain 실행 함수"""
    try:
        result = careerSummary_chain.invoke({
            "messages": backend_data
        })

        formatted_result = format_career_summary(result)

        user_info = backend_data.get("user_info", {})
        profile_id = user_info.get('profileId', '')
        
        
        return {
            "profile_id": profile_id,
            "career_summary": formatted_result
        }
    except Exception as e:
        return {
            "profile_id": "",
            "career_summary": f"오류 발생: {str(e)}"
        }

