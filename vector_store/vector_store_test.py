import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chroma_client import get_chroma_client

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
HISTORY_COLLECTION_NAME = os.getenv("HISTORY_COLLECTION_NAME")

# 환경 변수 확인
if not EMBEDDING_MODEL_NAME:
    print("오류: EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")
    exit(1)
if not HISTORY_COLLECTION_NAME:
    print("오류: HISTORY_COLLECTION_NAME 환경 변수가 설정되지 않았습니다.")
    exit(1)

def get_remote_vectordb():
    """원격 ChromaDB 연결"""
    # 임베딩 모델 준비
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
    )
    
    # 원격 ChromaDB 클라이언트 생성
    chroma_client = get_chroma_client()
    
    # LangChain Chroma로 원격 연결
    vectordb = Chroma(
        client=chroma_client,
        collection_name=HISTORY_COLLECTION_NAME,
        embedding_function=embeddings,
    )
    
    return vectordb

# def test_remote_connection():
#     """원격 ChromaDB 연결 테스트"""
#     try:
#         chroma_client = get_chroma_client()
#         collection = chroma_client.get_or_create_collection(name=HISTORY_COLLECTION_NAME)
#         print(f"✅ 원격 ChromaDB 연결 성공!")
#         print(f"컬렉션: {HISTORY_COLLECTION_NAME}")
#         print(f"저장된 문서 수: {collection.count()}")
#         return True
#     except Exception as e:
#         print(f"❌ 원격 ChromaDB 연결 실패: {e}")
#         return False

def search_by_profile_id(vectordb, profile_id):
    """1. 메타데이터 profileId로 검색해서 데이터 전체 가져오기"""
    print(f"\n=== 1. profileId={profile_id}로 검색 ===")
    
    try:
        # 메타데이터 필터로 검색
        docs = vectordb.get(where={"profileId": profile_id})
        
        if not docs['documents']:
            print(f"profileId={profile_id}에 해당하는 문서가 없습니다.")
            return None
        
        print(f"찾은 문서 수: {len(docs['documents'])}")
        
        for i, (doc_content, metadata) in enumerate(zip(docs['documents'], docs['metadatas'])):
            print(f"\n--- 문서 {i+1} ---")
            print(f"profileId: {metadata.get('profileId', 'N/A')}")
            print(f"employeeId: {metadata.get('employeeId', 'N/A')}")
            print(f"career_title: {metadata.get('career_title', 'N/A')}")
            print(f"내용: {doc_content[:500]}...")
            
        return docs
        
    except Exception as e:
        print(f"메타데이터 검색 중 오류: {e}")
        return None

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

def get_all_profile_ids(vectordb):
    """저장된 모든 profileId 확인"""
    try:
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name=HISTORY_COLLECTION_NAME)
        
        # 모든 메타데이터 가져오기
        all_data = collection.get()
        profile_ids = set()
        
        for metadata in all_data['metadatas']:
            if 'profileId' in metadata:
                profile_ids.add(metadata['profileId'])
        
        return sorted(list(profile_ids))
        
    except Exception as e:
        print(f"profileId 목록 가져오기 실패: {e}")
        return []

def test_remote_connection():
    """원격 ChromaDB 연결 테스트 & nav7 DB 확인"""
    try:
        chroma_client = get_chroma_client()

        # 1) 클라이언트가 쓰는 database 속성 찍어 보기
        db_attr = getattr(chroma_client, "_database", None) or getattr(chroma_client, "database", None)
        print("▶ 클라이언트에 설정된 database:", db_attr)

        # 2) v2 컬렉션 리스트 가져오기
        coll_list = chroma_client.list_collections()
        names     = [c.name for c in coll_list]     # ✨ 속성 접근
        print("▶ 현재 DB에 있는 컬렉션 목록:", names)  # names 변수 출력

        # 3) 네가 쓰려는 컬렉션이 진짜 있나 확인
        if HISTORY_COLLECTION_NAME in names:        # names 리스트로만 검사
            print(f"✅ 컬렉션 `{HISTORY_COLLECTION_NAME}` 이(가) 존재합니다.")
        else:
            print(f"⚠️ 컬렉션 `{HISTORY_COLLECTION_NAME}` 이(가) DB에 없습니다.")

        # 4) 컬렉션 카운트 확인
        collection = chroma_client.get_or_create_collection(name=HISTORY_COLLECTION_NAME)
        count = collection.count().get("count")
        print(f"▶ `{HISTORY_COLLECTION_NAME}` 컬렉션 벡터 수:", count)

        return True
    except Exception as e:
        print("❌ 원격 ChromaDB 연결 오류:", e)
        return False


if __name__ == "__main__":
    print("=== 원격 ChromaDB 테스트 시작 ===")
    
    # 연결 테스트
    if not test_remote_connection():
        print("원격 연결 실패로 중단합니다.")
        exit(1)
    
    # ChromaDB 연결
    vectordb = get_remote_vectordb()
    print(f"✅ 원격 vectordb 연결 완료")
    
    # 저장된 profileId 목록 확인
    # profile_ids = get_all_profile_ids(vectordb)
    # print(f"\n저장된 profileId 목록: {profile_ids}")
    
    # 1. 메타데이터로 검색 테스트 (첫 번째 profileId 사용)
    test_profile_id = 1
    search_by_profile_id(vectordb, test_profile_id)
    print("="*60)
    
    # 2. 쿼리 검색 테스트
    test_queries = "금융 도메인에서 사업개발 PM 해본 사람"
    
    search_by_query(vectordb, test_queries, k=2)
    
    print("\n=== 원격 ChromaDB 테스트 완료 ===")