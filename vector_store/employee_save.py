# import os
# import glob
# from dotenv import load_dotenv
# from tqdm import tqdm
# import sys

# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document

# load_dotenv()


# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
# EMP_DATA_DIR = os.getenv("EMP_DATA_DIR")
# HISTORY_COLLECTION_NAME = os.getenv("HISTORY_COLLECTION_NAME")



# # 환경 변수 확인
# if not EMBEDDING_MODEL_NAME:
#     print("오류: EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")
#     sys.exit(1)
# if not VECTOR_DB_DIR:
#     print("오류: VECTOR_DB_DIR 환경 변수가 설정되지 않았습니다.")
#     sys.exit(1)
# if not EMP_DATA_DIR:
#     print("오류: EMP_DATA_DIR 환경 변수가 설정되지 않았습니다.")
#     sys.exit(1)
# if not HISTORY_COLLECTION_NAME:
#     print("오류: HISTORY_COLLECTION_NAME 환경 변수가 설정되지 않았습니다.")
#     sys.exit(1)

# def save_emp_docs():
#     # 1) 임베딩 모델 준비
#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
#     )

#     # 2) ChromaDB 컬렉션 준비 (수정된 부분)
#     vectordb = Chroma(
#         persist_directory=VECTOR_DB_DIR,
#         embedding_function=embeddings,
#         collection_name=HISTORY_COLLECTION_NAME
#     )
#     print(f"🔗 컬렉션 연결: {HISTORY_COLLECTION_NAME}")

#     # 3) 파일 확인
#     files = sorted(glob.glob(os.path.join(EMP_DATA_DIR, "*.txt")))
#     print(f"📁 처리할 파일: {len(files)}개")
    
#     if not files:
#         raise FileNotFoundError(f"No .txt files in {EMP_DATA_DIR}")

#     documents_to_add = []
    
#     # 4) 파일 처리
#     for idx, path in enumerate(files, start=1):
#         emp_id = os.path.splitext(os.path.basename(path))[0]
        
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read().strip()
        
#         if not text:
#             print(f"⚠️ 빈 파일 건너뛰기: {emp_id}")
#             continue

#         # LangChain Document 객체 생성
#         doc = Document(
#             page_content=text,
#             metadata={
#                 "profileId": idx,
#                 "employeeId": emp_id,
#                 "career_title": ""
#             }
#         )
#         documents_to_add.append(doc)

#     print(f"📝 추가/업데이트할 데이터: {len(documents_to_add)}개")

#     # 5) 데이터 추가 (LangChain Chroma의 add_documents 사용)
#     # 기존 ID가 있으면 자동으로 덮어쓰기됩니다.
#     if documents_to_add:
#         vectordb.add_documents(documents_to_add)
#         print(f"✅ {len(documents_to_add)}개 문서 처리 완료 (추가/업데이트)")
#     else:
#         print("추가할 문서가 없습니다.")
    

# if __name__ == "__main__":
#     print('===================================')
#     print(f"현재 작업 디렉토리: {os.getcwd()}")
#     print(f"VECTOR_DB_DIR: {VECTOR_DB_DIR}")
#     print(f"절대 경로: {os.path.abspath(VECTOR_DB_DIR)}")
#     print(f"EMP_DATA_DIR: {EMP_DATA_DIR}")
#     print(f"데이터 절대 경로: {os.path.abspath(EMP_DATA_DIR)}")
#     print('===================================')
#     print(f"VECTOR_DB_DIR 원본: '{os.getenv('VECTOR_DB_DIR')}'")
#     print(f".env 파일 경로: {os.path.abspath('.env')}")
#     save_emp_docs()
#     print("=== ChromaDB 저장 완료 ===")



import os
import glob
from dotenv import load_dotenv
from tqdm import tqdm
import sys

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from chroma_client import get_chroma_client  # 원격 클라이언트 가져오기

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMP_DATA_DIR = os.getenv("EMP_DATA_DIR")
HISTORY_COLLECTION_NAME = os.getenv("HISTORY_COLLECTION_NAME")

# 환경 변수 확인 (VECTOR_DB_DIR 제거)
if not EMBEDDING_MODEL_NAME:
    print("오류: EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)
if not EMP_DATA_DIR:
    print("오류: EMP_DATA_DIR 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)
if not HISTORY_COLLECTION_NAME:
    print("오류: HISTORY_COLLECTION_NAME 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)

def save_emp_docs():
    # 1) 임베딩 모델 준비
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
    )

    # 2) 원격 ChromaDB 클라이언트 생성
    chroma_client = get_chroma_client()
    
    # 3) LangChain Chroma로 원격 연결
    vectordb = Chroma(
        client=chroma_client,  # 원격 클라이언트 사용
        collection_name=HISTORY_COLLECTION_NAME,
        embedding_function=embeddings,
    )
    print(f"🔗 원격 컬렉션 연결: {HISTORY_COLLECTION_NAME}")

    # 4) 파일 확인
    files = sorted(glob.glob(os.path.join(EMP_DATA_DIR, "*.txt")))
    print(f"📁 처리할 파일: {len(files)}개")
    
    if not files:
        raise FileNotFoundError(f"No .txt files in {EMP_DATA_DIR}")

    documents_to_add = []
    
    # 5) 파일 처리
    for idx, path in enumerate(files, start=1):
        emp_id = os.path.splitext(os.path.basename(path))[0]
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if not text:
            print(f"⚠️ 빈 파일 건너뛰기: {emp_id}")
            continue

        # LangChain Document 객체 생성
        doc = Document(
            page_content=text,
            metadata={
                "profileId": idx,
                "employeeId": emp_id,
                "career_title": ""
            }
        )
        documents_to_add.append(doc)

    print(f"📝 추가/업데이트할 데이터: {len(documents_to_add)}개")

    # 6) 원격 ChromaDB에 데이터 추가
    if documents_to_add:
        vectordb.add_documents(documents_to_add)
        print(f"✅ 원격 ChromaDB에 {len(documents_to_add)}개 문서 저장 완료!")
    else:
        print("추가할 문서가 없습니다.")

def test_remote_connection():
    """원격 ChromaDB 연결 테스트"""
    try:
        chroma_client = get_chroma_client()
        # 컬렉션 생성/연결 테스트
        collection = chroma_client.get_or_create_collection(name=HISTORY_COLLECTION_NAME)
        print(f"✅ 원격 ChromaDB 연결 성공!")
        print(f"컬렉션: {HISTORY_COLLECTION_NAME}")
        print(f"저장된 문서 수: {collection.count()}")
        return True
    except Exception as e:
        print(f"❌ 원격 ChromaDB 연결 실패: {e}")
        return False

def reset_collection():
    """컬렉션 삭제 후 재생성"""
    chroma_client = get_chroma_client()
    
    # 컬렉션 삭제
    try:
        chroma_client.delete_collection(name=HISTORY_COLLECTION_NAME)
        print(f"✅ 컬렉션 {HISTORY_COLLECTION_NAME} 삭제 완료")
    except:
        print("컬렉션이 존재하지 않거나 이미 삭제됨")
    
    # 새 컬렉션 생성
    chroma_client.create_collection(name=HISTORY_COLLECTION_NAME)
    print(f"✅ 새 컬렉션 {HISTORY_COLLECTION_NAME} 생성 완료")

if __name__ == "__main__":
    if test_remote_connection():
        # 방법 1 또는 방법 2 중 선택
        reset_collection()  # 컬렉션 삭제 후 재생성
        # clear_collection_data()  # 데이터만 삭제
        
        save_emp_docs()
        print("=== 원격 ChromaDB 저장 완료 ===")

