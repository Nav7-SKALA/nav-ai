import os
import urllib.parse
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# ── 1) .env에서 환경변수 읽기 ────────────────────────────────────────
raw_host = os.getenv("CHROMA_URL_EXTERNAL", "").strip()
basic_auth = os.getenv("BASIC_AUTH_HEADER", "").strip()

if not raw_host or not basic_auth:
    raise ValueError("CHROMA_URL_EXTERNAL과 BASIC_AUTH_HEADER를 .env에 설정해주세요.")

# “https://”나 “http://”가 붙어 있어도 호스트네임만 뽑아내기
parsed = urllib.parse.urlparse(raw_host if raw_host.startswith("http") else "http://" + raw_host)
HOST = parsed.hostname  
# ────────────────────────────────────────────────────────────────────

# ── 2) Settings 객체 생성 (v2 API를 쓰도록 지정) ────────────────────
settings = Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    chroma_server_host=HOST,
    chroma_server_http_port=443,                # HTTPS 기본 포트
    chroma_server_headers={"Authorization": basic_auth},
)
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tenant = "default"         # 실제 존재하는 tenant 이름
    database = "my_database"   # 생성(또는 연결)하려는 database 이름
    collection_name = "my_collection"  # 생성할 컬렉션 이름

    # ── 3) tenant/database 지정하여 HttpClient 생성 ─────────────────
    #    database를 지정하지 않으면 내부적으로 “default”를 쓰지만,
    #    여기서는 my_database를 바로 지정합니다.
    client = chromadb.HttpClient(
        settings=settings,
        host=HOST,                 # Settings와 반드시 동일하게 지정
        port=443,
        ssl=True,
        headers={"Authorization": basic_auth},
        tenant="",             # tenant="default"
        #database=database,         # database="my_database"
    )

    # ── 4) Heartbeat으로 서버 연결 확인 ─────────────────────────────
    try:
        hb = client.heartbeat()
        print("✅ ChromaDB 연결 성공 (heartbeat):", hb)
    except Exception as e:
        print("❌ heartbeat 호출 실패:", e)
        exit(1)

    # # ── 5) 컬렉션 생성 또는 가져오기 ─────────────────────────────────
    # #    get_or_create_collection을 쓰면, database가 없으면 내부에서 자동으로 생성합니다.
    # try:
    #     collection = client.get_or_create_collection(collection_name)
    #     print(f"✅ 컬렉션 생성/가져오기 성공: {tenant}/{database}/{collection.name}")
    # except Exception as e:
    #     print("❌ 컬렉션 생성 실패:", e)
    #     exit(1)

    # # ── 6) 생성된 컬렉션 리스트 확인 ─────────────────────────────────
    # try:
    #     cols = client.list_collections()
    #     names = [col.name for col in cols]
    #     print(f"🔍 지금 연결된 [{tenant}/{database}]의 컬렉션 목록:", names)
    # except Exception as e:
    #     print("❌ 컬렉션 목록 조회 실패:", e)
    #     exit(1)
