import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from prompt import roleModel_prompt
from config import MODEL_NAME, TEMPERATURE
from role_model_info_node import search_by_query
from vector_store_test import search_by_profile_id #test용 함수 없어도됌

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

def search_candidates(vectordb, query, k=10):
    """쿼리와 유사한 인물 k명 검색"""
    try:
        docs_with_score = vectordb.similarity_search_with_score(query, k=k)
        
        if not docs_with_score:
            return []
        
        candidates = []
        for doc, score in docs_with_score:
            candidate = {
                'profile_id': doc.metadata.get('profileId'),
                'employee_id': doc.metadata.get('employeeId'),
                'career_title': doc.metadata.get('career_title', ''),
                'content': doc.page_content,
                'similarity_score': score
            }
            candidates.append(candidate)
        
        return candidates
        
    except Exception as e:
        print(f"검색 중 오류: {e}")
        return []

def format_candidates_info(candidates):
    """후보자 정보를 LLM 입력용으로 포맷팅"""
    info_text = ""
    for i, candidate in enumerate(candidates, 1):
        info_text += f"후보 {i}:\n"
        info_text += f"- Profile ID: {candidate['profile_id']}\n"
        info_text += f"- 사원번호: {candidate['employee_id']}\n"
        info_text += f"- 유사도 점수: {candidate['similarity_score']:.4f}\n"
        info_text += f"- 경력 제목: {candidate['career_title']}\n"
        info_text += f"- 경력 내용: {candidate['content']}...\n"
        info_text += f"{'='*50}\n\n"
    return info_text

def recommend_top3_rolemodels(vectordb, query):

    candidates = search_candidates(vectordb, query, k=10)

    if not candidates:
        return {"error": "추천할 롤모델이 없습니다."}
    
    print(f"검색된 후보자 수: {len(candidates)}명")
    
    # 2. 후보자 정보 포맷팅
    candidates_info = format_candidates_info(candidates)
    
    # 3. LLM을 통해 상위 3명 선택
    try:
        chain = rm_prompt | llm | JsonOutputParser()
        result = chain.invoke({"role_model_info": candidates_info})
        
        return {
            "query": query,
            "recommendation": result,
            "total_candidates": len(candidates)
        }
        
    except Exception as e:
        return {"error": f"추천 중 오류: {str(e)}"}

        
def main():
    """테스트 실행"""
    from role_model_info_node import get_remote_vectordb
    
    # 벡터DB 초기화
    vectordb = get_remote_vectordb()
    
    # 테스트 쿼리
    query = "금융 프로젝트를 해봤던 pm정보 알려줘"
    
    print(f"쿼리: {query}")
    print("="*60)
    
    result = recommend_top3_rolemodels(vectordb, query)
    
    if "error" in result:
        print(f"오류: {result['error']}")
    else:
        print("\n추천 결과:")
        # 추천받은 profileId들의 상세 정보 출력
        
    for recommended in result['recommendation']:
        print(recommended['profileId'])
        search_by_profile_id(vectordb, int(recommended['profileId']))

if __name__ == "__main__":
    main()