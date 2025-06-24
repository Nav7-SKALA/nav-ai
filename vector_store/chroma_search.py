import chromadb
import os
import re
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_chroma_client():
    """원격 ChromaDB 클라이언트 생성 (데이터베이스 포함)"""
    client = chromadb.HttpClient(
        host="chromadb-1.skala25a.project.skala-ai.com",
        port=443,
        ssl=True,
        headers={
            "Authorization": "Basic YWRtaW46U2thbGEyNWEhMjMk"
        },
        database="nav7"  # 데이터베이스 이름 지정
    )
    
    return client

def find_best_match(query_text: str, user_id: str):  # user_id 파라미터 추가
    """쿼리에 가장 적합한 인재 찾기"""
    
    # 1. 상위 5명 정보 가져오기
    candidates = get_top5_info(query_text, user_id)  # user_id 넘겨주기
    
    # 2. LLM으로 1명 선택
    llm_choice = llm_select(query_text, candidates)
    
    # 3. 선택된 사번으로 상세 정보 반환
    emp_id = re.search(r'EMP-\d+', llm_choice).group(0)
    return get_employee_detail(emp_id)

def get_top5_info(query_text, user_id):
    """상위 5명 간단 정보"""
    client = get_chroma_client()
    collection = client.get_collection(name=os.getenv("JSON_HISTORY_COLLECTION_NAME"))
    model_path="c:/Users/Administrator/Desktop/김현준/최종프로젝트/nav-ai/model/ko-sroberta-multitask"
    embedding_model = SentenceTransformer(model_path)
    # embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME"))
    query_embedding = embedding_model.encode([query_text]).tolist()
    
    results = collection.query(query_embeddings=query_embedding, n_results=20, include=['metadatas'])
    
    # 중복 제거로 5명 선택
    seen = set()
    top5 = []
    for meta in results['metadatas'][0]:
        emp_id = meta['사번']
        if emp_id not in seen and emp_id != user_id:
            seen.add(emp_id)
            top5.append(emp_id)
            if len(top5) == 5:
                break
    
    # 간단한 후보 정보
    info = ""
    for i, emp_id in enumerate(top5, 1):
        emp_data = collection.get(where={"사번": emp_id}, include=['metadatas'])
        meta = emp_data['metadatas'][0]
        info += f"{i}. {meta['사번']} - {meta['grade']} - {len(emp_data['metadatas'])}개 경력\n"
    
    return info

def llm_select(query_text, candidates):
    """LLM으로 1명 선택"""
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME"), temperature=0.3)
    
    prompt = PromptTemplate(
        input_variables=["query", "candidates"],
        template="""
    쿼리에 가장 적합한 인재 1명을 선택해주세요.

    **사용자 요청:**
    {query}

    **후보자들:**
    {candidates}

    **평가 기준:**
    1. 쿼리와의 관련성 (기술 스택, 경험, 역할 등)
    2. 경력 수준과 적합성
    3. 도메인 경험
    4. 성장 가능성 및 전환 가능성

    선택: [사번]
    이유: [간단한 선택 이유]
    """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query_text, "candidates": candidates})

def get_employee_detail(emp_id):
    """선택된 사원 상세 정보"""
    client = get_chroma_client()
    collection = client.get_collection(name=os.getenv("JSON_HISTORY_COLLECTION_NAME"))
    
    emp_data = collection.get(where={"사번": emp_id}, include=['metadatas', 'documents'])
    
    result = f"=== 선택된 사원 ===\n"
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
    
    return result