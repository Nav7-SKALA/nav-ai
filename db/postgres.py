import psycopg2
from psycopg2 import sql
import os
def connect_and_query():
    try:
        print(os.getenv("POSTGRES_URL"))
        # PostgreSQL 연결
        conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
        print("✅ 연결 성공")
        # 커서 생성
        cur = conn.cursor()

        # 예시 쿼리 실행
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print("PostgreSQL 버전:", version[0])

        # 연결 종료
        cur.close()
        conn.close()
        print("🔌 연결 종료")

    except Exception as e:
        print("❌ 에러 발생:", e)