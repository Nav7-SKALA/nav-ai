from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import os
from config import DOCS_DIR, DB_PATH


def ensure_employee_vector_db_exists(docs_directory=DOCS_DIR, career_titles=None, db_path=DB_PATH, collection_name="employee_documents"):
    """
    직원 벡터 DB가 없으면 생성
    """
    if os.path.exists(db_path) and os.path.isdir(db_path) and len(os.listdir(db_path)) > 0:
        print(f"✅ 직원 벡터 DB 이미 존재: {db_path}")
        # 기존 벡터스토어 로드
        embeddings = SentenceTransformerEmbeddings(model_name='jhgan/ko-sroberta-multitask')
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=db_path
        )
        return vectorstore

    print("📦 직원 벡터 DB 생성 시작")
    return build_employee_vector_db(docs_directory, career_titles, db_path, collection_name)


def build_employee_vector_db(docs_directory, career_titles=None, db_path=DB_PATH, collection_name="employee_documents"):
    """
    직원 문서를 LangChain Chroma에 저장 (강제 재생성)
    
    Args:
        docs_directory: 문서 디렉토리 경로
        career_titles: 사원번호:커리어타이틀 딕셔너리 (선택적)
        db_path: ChromaDB 저장 경로
        collection_name: 컬렉션 이름
    """
    if not os.path.exists(docs_directory):
        raise FileNotFoundError(f"문서 디렉토리가 존재하지 않습니다: {docs_directory}")
    
    # LangChain 임베딩 모델 로드
    embeddings = SentenceTransformerEmbeddings(model_name='jhgan/ko-sroberta-multitask')
    
    # 파일명 오름차순 정렬
    txt_files = [f for f in os.listdir(docs_directory) if f.endswith('.txt')]
    txt_files.sort()
    
    if not txt_files:
        raise FileNotFoundError(f"txt 파일이 없습니다: {docs_directory}")
    
    documents = []
    
    # 문서 처리
    for idx, filename in enumerate(txt_files, 1):
        file_path = os.path.join(docs_directory, filename)
        employee_id = filename.split('.')[0]  # EMP-202827 (확장자 제거)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 메타데이터 생성
        metadata = {
            "profileId": idx,
            'employee_id': employee_id,
            'career_title': career_titles.get(employee_id, "") if career_titles else ""
        }
        
        # Document 객체 생성
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        documents.append(doc)
    
    # 벡터스토어 생성 및 문서 추가
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=db_path
    )
    
    print(f"✅ 직원 벡터 DB 저장 완료: {db_path} (총 {len(documents)}개 문서)")
    return vectorstore

# # 사용 예시
# if __name__ == "__main__":
#     career_titles_dict = {
#         'EMP-198261': '시니어 컨설턴트',
#         'EMP-202827': '수석 PM'
#     }
    
#     vectorstore = ensure_employee_vector_db_exists(DOCS_DIR, career_titles_dict)
    
#     # 올바른 방법: vectorstore 객체의 메서드 직접 호출
#     scored_results = vectorstore.similarity_search_with_score(
#         "profileId가 1인 사람 찾아줘", 
#         k=3  # n_results가 아니라 k
#     )
    
#     for i, (doc, score) in enumerate(scored_results):
#         print(f"\n{i+1}번째 문서 (거리 점수: {score:.4f}):")
#         print(f"메타데이터: {doc.metadata}")
#         print(f"내용: {doc.page_content[:200]}...")