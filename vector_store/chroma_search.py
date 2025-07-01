
import os
import re
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chroma_client import get_chroma_client

from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def find_best_match(query_text: str, user_id: str):
    """쿼리에 가장 적합한 인재 찾기"""
    
    # 1. 상위 10명 정보 가져오기
    candidates = get_topN_info(query_text, user_id, 10)  # user_id 넘겨주기
    
    # 2. LLM으로 5명 선택
    emp_ids = llm_select(query_text, candidates)
    
    # 3. 선택된 사번으로 상세 정보 반환
    # emp_id = re.search(r'EMP-\d+', llm_choice).group(0)
    return get_multiple_employees_detail(emp_ids)


def get_topN_info(query_text, user_id, top_n):
    """(LLM 위한) 상위 n명 간단 정보"""
    client = get_chroma_client()
    collection_name = os.getenv("JSON_HISTORY_COLLECTION_NAME")
    collection = client.get_collection(name=collection_name)

    embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME"))
    query_embedding = embedding_model.encode([query_text]).tolist()
    
    results = collection.query(query_embeddings=query_embedding, n_results=20, include=['metadatas'])
    
    # 중복 제거로 n명 선택
    seen = set()
    topN = []
    for meta in results['metadatas'][0]:

        emp_id = meta['사번']
        if emp_id not in seen and emp_id != user_id:
            seen.add(emp_id)
            topN.append(emp_id)
            if len(topN) == top_n:

                break
    
    # 각 profileId별 전체 경력 정보 구성
    info = ""

    for i, profile_id in enumerate(topN, 1):
        # 해당 profileId의 모든 경력 데이터 가져오기
        emp_data = collection.get(where={"profileId": profile_id}, include=['metadatas'])
        
        if not emp_data['metadatas']:
            continue
            
        first_meta = emp_data['metadatas'][0]
        
        info += f"\n{i}. profileId: {profile_id}\n"
        info += f"   사번: {first_meta['사번']}\n"
        info += f"   Grade: {first_meta['grade']}\n"
        info += f"   입사년도: {first_meta['입사년도']}\n"
        info += f"   경력흐름:\n"
        
        # 연차순으로 정렬해서 경력 흐름 구성
        careers = sorted(emp_data['metadatas'], key=lambda x: x['연차'])
        
        for j, career in enumerate(careers, 1):
            info += f"     {j}. {career['연차']} - {career['역할']}\n"
            info += f"        스킬셋: {career['스킬셋']}\n"
            info += f"        도메인: {career['도메인']}\n"
            info += f"        프로젝트규모: {career['프로젝트규모']}\n"
            info += f"        요약: {career['요약']}\n"
        
        info += "-" * 50 + "\n"
    
    return info

def get_topN_emp(query_text, user_id, top_n):
    """(roleModel agent 위한) 상위 n명 추출 - 개선된 버전"""
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=os.getenv("JSON_HISTORY_COLLECTION_NAME"))
        embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME"))
        query_embedding = embedding_model.encode([query_text]).tolist()
        
        results = collection.query(query_embeddings=query_embedding, n_results=20, include=['metadatas'])
        
        # 검색 결과가 없는 경우 처리
        if not results['metadatas'] or not results['metadatas'][0]:
            print("검색 결과가 없습니다.")
            return ""
        
        # 중복 제거로 n명 선택
        seen = set()
        topN = []
        for meta in results['metadatas'][0]:
            emp_id = meta.get('사번')  # .get() 사용으로 KeyError 방지
            if emp_id and emp_id not in seen and emp_id != user_id:
                seen.add(emp_id)
                topN.append(emp_id)
                if len(topN) == top_n:
                    break
        
        # 선택된 직원이 없는 경우 처리
        if not topN:
            print(f"선택된 직원이 없습니다. (user_id: {user_id} 제외 후)")
            return ""
        
        # print(f"선택된 직원 {len(topN)}명: {topN}")
        
        # 상세 정보 가져오기
        info = get_multiple_employees_detail(topN)
        
        # 빈 결과인 경우 처리
        if not info or info.strip() == "" or info.strip() == "\n" + "="*50:
            print("상세 정보 조회 실패")
            return ""
            
        return info
        
    except Exception as e:
        print(f"get_topN_emp 오류: {e}")
        return ""


def llm_select(query_text, candidates):
    """LLM으로 5명 선택"""
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME"), temperature=0.3)
    
    prompt = PromptTemplate(
        input_variables=["query", "candidates"],
        template="""
    쿼리에 가장 적합한 인재 5명을 선택해주세요.

    **사용자 요청:**
    {query}

    **후보자들:**
    {candidates}

    **평가 기준:**
    1. 쿼리와의 관련성 (기술 스택, 경험, 역할 등)
    2. 경력 수준과 적합성
    3. 도메인 경험
    4. 성장 가능성 및 전환 가능성

    **출력 형식:**
    반드시 선택된 인재(ID)만 콤마를 기준으로 연결하여 출력하세요.

    예시: EMP001, EMP002, EMP003, EMP004, EMP005

    선택: [profileId]
    이유: [간단한 선택 이유]
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query_text, "candidates": candidates})
    
    ids = [id.strip() for id in result.strip().split(',')]
    return ids

def get_multiple_employees_detail(emp_ids):
    # TODO: experience, certification 추가해야 함
    """선택된 여러 사원들의 상세 정보를 합쳐서 반환"""
    client = get_chroma_client()
    collection = client.get_collection(name=os.getenv("JSON_HISTORY_COLLECTION_NAME"))
    
    all_results = []
    
    for emp_id in emp_ids:
        try:
            emp_data = collection.get(where={"사번": emp_id}, include=['metadatas', 'documents'])
            
            if not emp_data['metadatas']:  # 데이터가 없는 경우 스킵
                continue
                
            result = f"=== 선택된 사원 ({emp_id}) ===\n"
            first_meta = emp_data['metadatas'][0]
            result += f"사번: {first_meta['사번']}\n"
            result += f"Grade: {first_meta['grade']}\n"
            result += f"입사년도: {first_meta['입사년도']}\n"
            result += f"총 경력 수: {len(emp_data['metadatas'])}개\n\n"
            
            result += "경력 상세:\n"
            for j, (meta, doc) in enumerate(zip(emp_data['metadatas'], emp_data['documents']), 1):
                result += f"  경력 {j}: {meta['연차']} - {meta['역할']}\n"
                result += f"    스킬셋: {meta['스킬셋']}\n"
                result += f"    도메인: {meta['도메인']}\n"
                result += f"    프로젝트규모: {meta['프로젝트규모']}\n"
                result += f"    요약: {meta['요약']}\n"
                result += f"    상세내용: {doc}\n\n"
            
            all_results.append(result)
            
        except Exception as e:
            print(f"직원 {emp_id} 정보 조회 실패: {e}")
            continue
    
    # 모든 결과를 하나의 문자열로 합치기
    combined_result = "\n" + ("="*50 + "\n").join(all_results)
    return combined_result

def get_employee_detail(profile_id):
    """선택된 사원 상세 정보"""
    client = get_chroma_client()
    collection = client.get_collection(name=os.getenv("JSON_HISTORY_COLLECTION_NAME"))
    
    emp_data = collection.get(where={"profileId": profile_id}, include=['metadatas', 'documents'])

    if not emp_data['metadatas']:
        return f"profileId '{profile_id}'를 찾을 수 없습니다."
    
    result = f"=== 선택된 사원 ===\n"
    first_meta = emp_data['metadatas'][0]
    result += f"profileId: {first_meta['profileId']}\n"
    result += f"사번: {first_meta['사번']}\n"
    result += f"Grade: {first_meta['grade']}\n"
    result += f"입사년도: {first_meta['입사년도']}\n"
    result += f"총 경력 수: {len(emp_data['metadatas'])}개\n\n"
    
    result += "경력 상세:\n"
    for j, (meta, doc) in enumerate(zip(emp_data['metadatas'], emp_data['documents']), 1):
        result += f"  경력 {j}: {meta['연차']} - {meta['역할']}\n"
        result += f"    스킬셋: {meta['스킬셋']}\n"
        result += f"    도메인: {meta['도메인']}\n"
        result += f"    프로젝트규모: {meta['프로젝트규모']}\n"
        result += f"    요약: {meta['요약']}\n"
        result += f"    상세내용: {doc}\n\n"
    
    return result

if __name__ == "__main__":
    result = find_best_match("금융프로젝트 pm", user_id=1)
    print(result)
