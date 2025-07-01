import psycopg2
from psycopg2 import sql
import os

def get_career_summary(member_id: str):
    if not member_id.isdigit():
        return "None"
    # PostgreSQL 연결
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    # 커서 생성
    cur = conn.cursor()
    # 예시 쿼리 실행
    cur.execute("""
        SELECT career_summary
        FROM public.profile
        WHERE member_id = %s;
    """, (member_id,))
    result = cur.fetchone()
    # 연결 종료
    cur.close()
    conn.close()
    if result is None:
        return "None"
    return result[0]
