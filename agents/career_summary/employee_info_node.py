import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage

from config import BASE_DIR
import sys
sys.path.append(BASE_DIR)
from vector_store.chroma_client import get_chroma_client

import warnings
# 허깅페이스 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
HISTORY_COLLECTION_NAME = os.getenv("HISTORY_COLLECTION_NAME")

def get_remote_vectordb():
    """원격 ChromaDB 연결"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
    )
    
    chroma_client = get_chroma_client()
    vectordb = Chroma(
        client=chroma_client,
        collection_name=HISTORY_COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vectordb

def search_profile_by_id(state):
    """사원번호 또는 profileId로 프로필 검색"""
    try:
        # 입력값 추출
        employee_id = state.get("employee_id")
        profile_id = state.get("profile_id")
        
        if not employee_id and not profile_id:
            return {"messages": [AIMessage(content="employee_id 또는 profile_id를 입력해주세요.")]}
        
        # VectorDB 연결 및 검색
        vectordb = get_remote_vectordb()
        
        if employee_id:
            docs = vectordb.get(where={"employeeId": employee_id})
            search_info = f"사원번호 {employee_id}"
        else:
            docs = vectordb.get(where={"profileId": int(profile_id)})
            search_info = f"profileId {profile_id}"
        
        if not docs['documents']:
            return {"messages": [AIMessage(content=f"{search_info}에 해당하는 정보가 없습니다.")]}
        
        # 결과 정리
        profile_data = []
        for doc_content, metadata in zip(docs['documents'], docs['metadatas']):
            profile_data.append({
                "profileId": metadata.get('profileId'),
                "employeeId": metadata.get('employeeId'),
                "career_title": metadata.get('career_title'),
                "content": doc_content
            })
        
        content = f"✅ {search_info} 정보 검색 완료 (총 {len(profile_data)}개 문서)"
        
        return {
            "messages": [AIMessage(content=content)],
            "profile_data": profile_data
        }
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"검색 중 오류: {str(e)}")]}

# if __name__ == "__main__":

#     vectordb = get_remote_vectordb()
#     print(f"✅ 원격 vectordb 연결 완료")

#     # 사원번호로 테스트
#     state = {"employee_id": "EMP-198261"}
#     result = search_profile_by_id(state)
#     print(result)
    
    # 또는 profileId로 테스트  
    # state = {"profile_id": "1"}
    # result = search_profile_by_id(state)
    # print(result)