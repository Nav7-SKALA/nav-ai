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

def get_company_direction():
    # PostgreSQL 연결
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    # 커서 생성
    cur = conn.cursor()
    try:    
        # 최근 행 출력 쿼리 실행
        cur.execute("""
            SELECT prompt 
            FROM public.direction 
            ORDER BY created_at DESC 
            LIMIT 1;
        """)
        result = cur.fetchone()
    except:
        # 하드코딩 버전
        result = ['모든 업무와 프로젝트에 AI를 기본 적용할 줄 아는 AI 기본 역량을 갖춘 사내 구성원']
    # 연결 종료
    cur.close()
    conn.close()
    return result[0]