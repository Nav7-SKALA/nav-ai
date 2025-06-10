import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import roleModel_prompt
from config import MODEL_NAME, TEMPERATURE
from role_model_info_node import search_by_query  # 임포트 변경

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# RoleModel 프롬프트
rm_prompt = PromptTemplate(
    input_variables=["role_model_info"],
    template=roleModel_prompt
)

# RoleModel 체인
roleModel_chain = rm_prompt | llm | StrOutputParser()

def role_model_node(state):
    """롤모델 추천 노드 - search_results를 받아서 추천"""
    try:
        # role_model_info_node에서 가져온 search_results 추출
        search_results = state.get("search_results", [])
        
        if not search_results:
            return {
                **state,
                "messages": state.get("messages", []) + [AIMessage(content="추천할 롤모델이 없습니다.")]
            }
        
        # search_results의 모든 정보를 텍스트로 변환
        results_text = ""
        for i, data in enumerate(search_results, 1):
            results_text += f"후보 {i}:\n"
            results_text += f"Profile ID: {data.get('profile_id')}\n"
            results_text += f"사원번호: {data.get('employeeId')}\n"
            results_text += f"유사도 점수: {data.get('similarity_score')}\n"
            results_text += f"경력 내용: {data.get('content', '')[:500]}...\n"  # 내용 일부만
            results_text += f"{'='*50}\n\n"
        
        # 롤모델 추천 실행 - employee_info로 전달
        result = roleModel_chain.invoke({"role_model_info": results_text})
        
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content=result)]
        }
        
    except Exception as e:
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content=f"롤모델 추천 중 오류: {str(e)}")]
        }


if __name__ == "__main__":
    # 테스트
    test_state = {
        "search_results": [
            {"profile_id": 1, "similarity_score": 0.85},
            {"profile_id": 5, "similarity_score": 0.78}
        ]
    }
    result = role_model_node(test_state)
    print(result["messages"][-1].content)
