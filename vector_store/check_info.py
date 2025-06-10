from chroma_client import get_chroma_client
import requests

def get_all_databases():
    """모든 데이터베이스 정보 가져오기"""
    response = requests.get(
        "https://chromadb-1.skala25a.project.skala-ai.com/api/v1/databases",
        headers={"Authorization": "Basic YWRtaW46U2thbGEyNWEhMjMk"}
    )
    
    if response.status_code == 200:
        databases = response.json()
        print("🗃️  모든 데이터베이스:")
        for db in databases:
            print(f"  - {db}")
        return databases
    else:
        print(f"❌ 데이터베이스 조회 실패: {response.status_code}")
        return None

def get_employee_documents_info():
    """employee_documents 컬렉션 상세 정보"""
    try:
        client = get_chroma_client()
        collection = client.get_collection("employee_documents")
        
        print("\n📋 employee_documents 컬렉션 정보:")
        print(f"  문서 수: {collection.count()}")
        
        # 첫 3개 문서 미리보기
        if collection.count() > 0:
            docs = collection.peek(limit=3)
            print(f"  미리보기 (첫 3개):")
            for i, (doc, metadata) in enumerate(zip(docs['documents'], docs['metadatas'])):
                print(f"    문서 {i+1}: profileId={metadata.get('profileId')}, employeeId={metadata.get('employeeId')}")
                print(f"            내용: {doc[:100]}...")
        
    except Exception as e:
        print(f"❌ employee_documents 조회 실패: {e}")

def check_chroma_info():
    """ChromaDB 전체 정보 확인"""
    
    try:
        client = get_chroma_client()
        
        # 1. 연결 확인
        client.heartbeat()
        print("✅ ChromaDB 연결 성공!")
        
        # 2. 버전 정보
        version = client.get_version()
        print(f"📋 ChromaDB 버전: {version}")
        
        # 3. 모든 데이터베이스 목록
        get_all_databases()
        
        # 4. 컬렉션 목록
        collections = client.list_collections()
        print(f"\n📁 현재 데이터베이스의 컬렉션:")
        for col in collections:
            count = col.count()
            print(f"  - {col.name}: {count}개 문서")
        
        # 5. employee_documents 상세 정보
        get_employee_documents_info()
            
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    check_chroma_info()