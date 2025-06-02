import os
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
# from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio # asyncio 임포트

from dotenv import load_dotenv
load_dotenv()

model_name = os.getenv("EMBEDDING_MODEL_NAME")
vector_db_dir = os.getenv("VECTOR_DB_DIR")


### test
@tool
def vectorDB_search(query: str) -> str:
    """vectorDB에서 정보 검색 도구"""
    return {"similar_id": ["1", "2", "3"]}

### real
# class KoreanEmbeddingModel:
#     def __init__(self, model_name=model_name): # .env에서 모델 이름 가져오도록 변경
#         """한국어 임베딩 모델 초기화
        
#         Args:
#             model_name (str): 사용할 임베딩 모델 이름
#         """
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)
    
#     def embed_query(self, query):
#         """쿼리 텍스트를 임베딩 벡터로 변환
        
#         Args:
#             query (str): 임베딩할 쿼리 텍스트
            
#         Returns:
#             numpy.ndarray: 임베딩 벡터
#         """
#         return self.model.encode(query)


# class VectorDBSearcher:
#     def __init__(self, persist_directory=vector_db_dir, embedding_model=None):
#         """벡터 DB 검색기 초기화
        
#         Args:
#             persist_directory (str): ChromaDB 저장 디렉토리 경로
#             embedding_model: 임베딩 모델 객체
#         """
#         self.persist_directory = persist_directory
#         self.embedding_model = embedding_model or KoreanEmbeddingModel()
        
#         # 벡터 DB 디렉토리가 존재하는지 확인
#         if not os.path.exists(persist_directory):
#             os.makedirs(persist_directory, exist_ok=True)
#             print(f"경고: {persist_directory} 디렉토리가 존재하지 않아 새로 생성했습니다.")
#             print("벡터 DB를 먼저 구축해야 검색이 가능합니다.")
#             self.vectordb = None
#         else:
#             try:
#                 # 커스텀 임베딩 함수 정의
#                 embedding_function = lambda texts: self.embedding_model.model.encode(texts)
                
#                 # ChromaDB 로드
#                 self.vectordb = Chroma(
#                     persist_directory=persist_directory,
#                     embedding_function=embedding_function
#                 )
#             except Exception as e:
#                 print(f"벡터 DB 로드 중 오류 발생: {e}")
#                 self.vectordb = None
    
#     def search_by_similarity(self, query, top_k=5):
#         """쿼리와 유사한 상위 k개 문서 검색
        
#         Args:
#             query (str): 검색 쿼리
#             top_k (int): 반환할 결과 개수
            
#         Returns:
#             list: 검색 결과 리스트 (각 항목은 문서 내용과 메타데이터 포함)
#         """
#         if self.vectordb is None:
#             return {"error": "벡터 DB가 초기화되지 않았습니다."}
        
#         # 쿼리 임베딩
#         query_embedding = self.embedding_model.embed_query(query)
        
#         # 유사도 검색 수행
#         results = self.vectordb.similarity_search_by_vector(
#             embedding=query_embedding,
#             k=top_k
#         )
        
#         # 결과 포맷팅
#         formatted_results = []
#         for i, doc in enumerate(results):
#             formatted_results.append({
#                 "rank": i + 1,
#                 "content": doc.page_content,
#                 "metadata": doc.metadata,
#                 "score": None  # ChromaDB는 기본적으로 점수를 반환하지 않음
#             })
        
#         return formatted_results


# # LangChain 도구로 등록
# @tool("search_vector_db", return_direct=True)
# def search_vector_db(query: str, top_k: int = 5, persist_directory: str = vector_db_dir):
#     """쿼리와 유사한 상위 k개 문서를 벡터 DB에서 검색합니다.
    
#     Args:
#         query (str): 검색할 쿼리 텍스트
#         top_k (int): 반환할 결과 개수 (기본값: 5)
#         persist_directory (str): 벡터 DB 저장 경로 (기본값: ./vector_store)
        
#     Returns:
#         list: 검색 결과 리스트
#     """
#     # 임베딩 모델 초기화
#     embedding_model = KoreanEmbeddingModel()
    
#     # 벡터 DB 검색기 초기화
#     searcher = VectorDBSearcher(
#         persist_directory=persist_directory,
#         embedding_model=embedding_model
#     )
    
#     # 검색 수행
#     results = searcher.search_by_similarity(query, top_k=top_k)
    
#     return results


# 사용 예시
# if __name__ == "__main__":
#     # 테스트 쿼리
#     test_query = "React와 Node.js 경험이 있는 개발자"
    
#     # 검색 수행
#     results = search_vector_db(test_query)
    
#     # 결과 출력
#     print(f"쿼리: {test_query}")
#     print(f"검색 결과 ({len(results)}개):")
#     for result in results:
#         print(f"순위 {result['rank']}: {result['content'][:100]}...")
#         print(f"메타데이터: {result['metadata']}")
#         print("-" * 50)