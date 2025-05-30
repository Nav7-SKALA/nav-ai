import os
from sqlalchemy import create_engine, text
import pandas as pd
from langchain.tools import tool
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class PostgreSQLConnector:
    def __init__(self):
        self.engine = self._create_engine()

    def _create_engine(self):
        # 환경 변수에서 데이터베이스 연결 정보 로드
        db_user = os.getenv("POSTGRES_USER")
        db_password = os.getenv("POSTGRES_PASSWORD")
        db_host = os.getenv("POSTGRES_HOST")
        db_port = os.getenv("POSTGRES_PORT")
        db_name = os.getenv("POSTGRES_DB")

        if not all([db_user, db_password, db_host, db_port, db_name]):
            raise ValueError("PostgreSQL connection environment variables are not set.")

        # SQLAlchemy 엔진 생성
        DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return create_engine(DATABASE_URL)

    def execute_query(self, query: str) -> str:
        """SQL 쿼리를 동기적으로 실행 후 결과 반환"""
        with self.engine.connect() as connection:
            try:
                result = connection.execute(text(query))
                # SELECT 쿼리인 경우 fetchall 사용
                if query.strip().upper().startswith("SELECT"):
                    # 결과를 문자열로 변환하여 반환
                    rows = result.fetchall()
                    return str(rows)
                else:
                    # INSERT, UPDATE, DELETE 등의 쿼리인 경우
                    connection.commit()
                    return f"Query executed successfully. Rows affected: {result.rowcount}"
            except Exception as e:
                return f"Error executing query: {e}"

    def execute_query_to_dataframe(self, query: str) -> pd.DataFrame:
        """SQL 쿼리를 동기적으로 실행 후 결과 Pandas DataFrame으로 반환"""
        with self.engine.connect() as connection:
            try:
                df = pd.read_sql_query(text(query), connection)
                return df
            except Exception as e:
                print(f"Error executing query to DataFrame: {e}")
                return pd.DataFrame()

# LangChain 도구로 사용할 함수 정의
connector = PostgreSQLConnector()

@tool
def query_postgres(query: str) -> str:
    """PostgreSQL 데이터베이스에 SQL 쿼리를 실행하고 결과를 문자열로 반환
    예시: SELECT * FROM users LIMIT 10;"""
    result = connector.execute_query(query)
    return result

@tool
def postgres_to_dataframe(query: str) -> str:
    """PostgreSQL 데이터베이스에 SQL 쿼리를 실행하고 결과를 Pandas DataFrame으로 반환
    결과는 문자열 형태로 변환되어 반환됩니다.
    예시: SELECT id, name FROM products WHERE price > 100;"""
    df = connector.execute_query_to_dataframe(query)
    # DataFrame을 문자열로 변환하여 반환
    return df.to_string()

@tool
def get_member_profile_details(member_id: int) -> str:
    """특정 멤버 ID가 입력되면, 해당 멤버의 프로젝트, 경험, 기술 스택, 자격증 정보를 동기적으로 조회, 출력

    Args:
        member_id (int): 정보를 조회할 멤버의 고유 ID.

    Returns:
        str: 조회된 멤버의 프로젝트, 경험, 기술 스택, 자격증 정보를 포함하는 문자열.
             정보가 없을 경우 해당 섹션은 빈 문자열 반환
    """
    results_str = f"### 멤버 ID: {member_id} 정보 ###\n\n"

    # 1. 프로젝트 정보 동기 조회
    project_query = f"""
    SELECT \"프로젝트 명\", \"프로젝트 묘사\", \"프로젝트 시작\", \"프로젝트 끝\"
    FROM \"멤버 프로젝트\"
    WHERE \"FK\" = {member_id};
    """
    projects_df = connector.execute_query_to_dataframe(project_query)
    results_str += "#### 프로젝트 정보 ####\n"
    if not projects_df.empty:
        results_str += projects_df.to_string(index=False) + "\n\n"
    else:
        results_str += "프로젝트 정보가 없습니다.\n\n"

    # 2. 경험 정보 동기 조회
    experience_query = f"""
    SELECT \"경험 이름\", \"경험 묘사\", \"경험한 날짜\"
    FROM \"멤버 경험\"
    WHERE \"FK\" = {member_id};
    """
    experiences_df = connector.execute_query_to_dataframe(experience_query)
    results_str += "#### 경험 정보 ####\n"
    if not experiences_df.empty:
        results_str += experiences_df.to_string(index=False) + "\n\n"
    else:
        results_str += "경험 정보가 없습니다.\n\n"

    # 3. 기술 스택 정보 동기 조회 (멤버 프로젝트와 기술 스택 테이블 조인)
    tech_stack_query = f"""
    SELECT DISTINCT T2.\"기술 스택 이름\"
    FROM \"프로젝트 기술 스택\" AS T1
    JOIN \"기술 스택\" AS T2 ON T1.\"PK3\" = T2.\"PK\"
    JOIN \"멤버 프로젝트\" AS T3 ON T1.\"PK\" = T3.\"PK\"
    WHERE T3.\"FK\" = {member_id};
    """
    tech_stack_df = connector.execute_query_to_dataframe(tech_stack_query)
    results_str += "#### 기술 스택 정보 ####\n"
    if not tech_stack_df.empty:
        results_str += tech_stack_df.to_string(index=False) + "\n\n"
    else:
        results_str += "기술 스택 정보가 없습니다.\n\n"

    # 4. 자격증 정보 동기 조회 (멤버 자격증과 자격증 테이블 조인)
    certification_query = f"""
    SELECT T2.\"자격증 이름\", T1.\"취득일\"
    FROM \"멤버 자격증\" AS T1
    JOIN \"자격증\" AS T2 ON T1.\"FK3\" = T2.\"PK\"
    WHERE T1.\"FK2\" = {member_id};
    """
    certifications_df = connector.execute_query_to_dataframe(certification_query)
    results_str += "#### 자격증 정보 ####\n"
    if not certifications_df.empty:
        results_str += certifications_df.to_string(index=False) + "\n\n"
    else:
        results_str += "자격증 정보가 없습니다.\n\n"

    return results_str

# 동기 도구를 실행하기 위한 예시
if __name__ == "__main__":
    member_id_to_search = 1 # 예시 멤버 ID
    profile_details = get_member_profile_details(member_id_to_search)
    print(profile_details)

    print("\n--- 일반 쿼리 테스트 ---")
    print(query_postgres("SELECT * FROM \"멤버 프로젝트\" LIMIT 2;"))

    print("\n--- DataFrame 쿼리 테스트 ---")
    df_test = postgres_to_dataframe("SELECT \"프로젝트 명\" FROM \"멤버 프로젝트\" LIMIT 2;")
    print(df_test)