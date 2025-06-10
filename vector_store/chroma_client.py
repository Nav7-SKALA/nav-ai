import chromadb

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

    
# # 로컬 chroma 서버에 연결하는 함수
# import chromadb
# from chromadb.config import Settings

# def get_chroma_client(persist_directory):
#     return chromadb.PersistentClient(
#         path=persist_directory
#     )
