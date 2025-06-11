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

def search_by_query(vectordb, query, k=2):
    """2. 쿼리 날려서 top2 데이터 가져오기"""
    print(f"\n=== 2. 쿼리 검색: '{query}' (상위 {k}개) ===")
    
    try:
        # 유사도 검색 (점수 포함)
        docs_with_score = vectordb.similarity_search_with_score(query, k=k)
        
        if not docs_with_score:
            print("검색 결과가 없습니다.")
            return None
        
        print(f"검색 결과: {len(docs_with_score)}개")
        
        for i, (doc, score) in enumerate(docs_with_score, 1):
            print(f"\n--- 결과 {i} (유사도 점수: {score:.4f}) ---")
            print(f"profileId: {doc.metadata.get('profileId', 'N/A')}")
            print(f"employeeId: {doc.metadata.get('employeeId', 'N/A')}")
            print(f"career_title: {doc.metadata.get('career_title', 'N/A')}")
            print(f"내용: {doc.page_content[:300]}...")
            
        return docs_with_score
        
    except Exception as e:
        print(f"쿼리 검색 중 오류: {e}")
        return None