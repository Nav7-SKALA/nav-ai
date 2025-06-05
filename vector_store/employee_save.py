import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

def create_vector_db(docs_directory, db_path="./chroma_db", collection_name="employee_documents"):
    """
    직원 문서를 ChromaDB에 저장
    
    Args:
        docs_directory: 문서 디렉토리 경로
        db_path: ChromaDB 저장 경로
        collection_name: 컬렉션 이름
    """
    # 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    # ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(path=db_path)
    
    # 기존 컬렉션이 있으면 삭제하고 새로 만들기
    try:
        client.delete_collection(collection_name)  # 기존 컬렉션 삭제
    except:
        pass  # 컬렉션이 없으면 무시
        
    collection = client.create_collection(collection_name)  # 새 컬렉션 생성
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    # 문서 처리
    for filename in os.listdir(docs_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(docs_directory, filename)
            employee_id = filename.split('_')[0]  # EMP-202827
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 각 프로젝트별로 분리
            projects = content.split('\n\n')
            
            for i, project in enumerate(projects):
                if project.strip():
                    # 고유 ID 생성
                    doc_id = f"{employee_id}_project_{i}"
                    
                    # 메타데이터 생성
                    metadata = {
                        'employee_id': employee_id,
                        'project_index': i
                    }
                    
                    # 커리어타이틀이 있다면 추가 (예시)
                    # metadata['career_title'] = get_career_title(employee_id)
                    
                    all_documents.append(project.strip())
                    all_metadatas.append(metadata)
                    all_ids.append(doc_id)
    
    # 임베딩 생성
    embeddings = model.encode(all_documents)
    
    # ChromaDB에 저장
    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_documents,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"총 {len(all_documents)}개 문서를 벡터 DB에 저장했습니다.")
    return collection

def search_documents(collection, query, model, n_results=5):
    """
    벡터 DB에서 문서 검색
    
    Args:
        collection: ChromaDB 컬렉션
        query: 검색 쿼리
        model: 임베딩 모델
        n_results: 반환할 결과 수
    """
    query_embedding = model.encode([query])
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    return results

# 사용 예시
if __name__ == "__main__":
    # 벡터 DB 생성
    collection = create_vector_db("emp_docs")
    
    # 검색용 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    # 검색 테스트
    results = search_documents(collection, "금융 도메인 PM", model, n_results=3)
    
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n{i+1}. {doc}")
        print(f"메타데이터: {results['metadatas'][0][i]}")