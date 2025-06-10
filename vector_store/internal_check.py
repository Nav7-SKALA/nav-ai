import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
LEC_COLLECTION_NAME = os.getenv("LEC_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

if not VECTOR_DB_DIR or not LEC_COLLECTION_NAME or not EMBEDDING_MODEL_NAME:
    print("환경 변수 (VECTOR_DB_DIR, LEC_COLLECTION_NAME, EMBEDDING_MODEL_NAME)가 제대로 설정되지 않았습니다.")
    exit()

# ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

# 임베딩 모델 로드
# HuggingFaceEmbeddings 초기화 시 encode_kwargs에 clean_up_tokenization_spaces=False 추가
# 이는 transformers 라이브러리의 FutureWarning를 해결하기 위함입니다.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
)

# ChromaDB 컬렉션 로드
# embedding_function 인자에 HuggingFaceEmbeddings 객체를 직접 전달
vectordb = Chroma(
    client=client,
    collection_name=LEC_COLLECTION_NAME,
    embedding_function=embeddings,
)

# 4) 유사도 검색 예시
if __name__ == "__main__":
    # (1) 검색할 질의(query) 문장 정의
    query_text = "AI 초심자 맞춤 강의"

    # query_embedding은 vectordb.similarity_search_with_score 또는 vectordb.similarity_search_by_vector를 사용할 경우 필요합니다.
    # 여기서는 vectordb.similarity_search를 사용하므로 별도로 임베딩할 필요가 없습니다.

    # (2) 유사도 검색 수행
    # LangChain의 Chroma 통합은 query_text를 직접 받아 내부적으로 임베딩을 수행합니다.
    results = vectordb.similarity_search_with_score(
        query=query_text,
        k=5 # 상위 5개 반환
    )

    print(f"\n'{query_text}'에 대한 유사도 검색 결과 (상위 5개):\n")
    for doc, score in results:
        print(f"문서 내용: {doc.page_content}\n메타데이터: {doc.metadata}\n유사도 거리: {score:.4f}\n---\n")

    print("유사도 검색이 완료되었습니다.")