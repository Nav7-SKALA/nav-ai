import os
import re
import pandas as pd
import openpyxl
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer


# 환경변수 로드
load_dotenv()

def read_excel_file(file_path, sheet_name=0):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # print(f"'{file_path}' 파일의 '{sheet_name}' 시트를 성공적으로 읽었습니다.")
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return None
    except Exception as e:
        print(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None


def build_skill_set_sentences(learning_guide):
    """
    learning_guide: learning_guide 칼럼의 행 하나
    
    반환값: 여러 줄로 된 문자열, 각 줄은 "<skill_type> skill set은 ...입니다." 형태
    """

    # 1) parse
    parts = re.split(r',(?=(?:특화|추천|공통필수)\s*-)', learning_guide)

    parsed = []
    for part in parts:
        key, domain, jobs_str = [x.strip() for x in part.split(" - ", maxsplit=2)]

        jobs_list = [job.strip() for job in jobs_str.split(",") if job.strip()]
        
        parsed.append({
            "key": key,            
            "domain": domain,      
            "jobs": jobs_list      
        })

    skill_types = ['특화', '추천', '공통필수']

    # 2) skill_types 별로 도메인 → 직무 리스트 묶기
    grouped = {stype: defaultdict(list) for stype in skill_types}
    for item in parsed:
        k = item.get('key')
        if k in grouped:
            domain = item['domain']
            grouped[k][domain].extend(item['jobs'])

    # 3) 각 skill_type에 대해 "도메인 직무의 직무1와 직무2, ..." 형태로 조합
    lines = []
    for stype in skill_types:
        domain_dict = grouped.get(stype, {})
        if not domain_dict:
            continue  # 해당 타입에 데이터가 없으면 건너뜀

        parts = []
        for domain, jobs in domain_dict.items():
            if not jobs:
                continue
            if len(jobs) == 1:
                parts.append(f"{domain} 직무의 {jobs[0]}")
            else:
                parts.append(f"{domain} 직무의 " + "와 ".join(jobs))

        joined = ", ".join(parts)
        lines.append(f"{stype} skill set은 {joined}입니다.")

    # 4) 줄바꿈으로 연결하여 반환
    return "\n".join(lines)


def trans_to_document(data_row):
    """입력된 한 행 데이터를 저장할 document text로 변환"""

    if pd.isna(data_row['Learning Guide']):
        skill_parse = ''
    else:
        skill_parse = build_skill_set_sentences(data_row['Learning Guide'])
            
    content_text = (
            f"**{data_row['교육과정명']}** 강의입니다. "
            f"이 과정은 {data_row['학부']} 학부의 {data_row['표준과정']} 표준 과정이며, "
            f"{data_row['교육유형']} 유형의 {data_row['학습유형']} 수업입니다."
            f"총 학습 시간은 {data_row['학습시간']}시간입니다."
            f"{skill_parse}"
    )
    return content_text


def create_chromadb_local(data, db_name, embed_model, persist_directory):
    """로컬 저장소(persist_directory)에 chromaDB 생성, 저장"""

    model = SentenceTransformer(embed_model)

    # db 구성
    client = chromadb.PersistentClient(
        path=persist_directory 
    )
    collection = client.get_or_create_collection(db_name)

    for i in tqdm(range(len(data))):
        row = data.iloc[i]

        content_text = trans_to_document(row)

        metadata = {
            '강의명': row['교육과정명'],
            "학부": row['학부'],
            "표준과정": row['표준과정'],
            "교육유형": row['교육유형'],
            "학습유형": row['학습유형'],
            "학습시간": row['학습시간']
        }

        # 임베딩 계산 후 저장
        embedding_vector = model.encode(content_text).tolist()
        
        collection.add(
            documents=[content_text],
            metadatas=[metadata],
            embeddings=embedding_vector,
            ids=[f"LEC_{i+1}"]
        )


if __name__ == "__main__":
    # 사내 강의 데이터 로컬 경로에 저장하는 초기화 코드

    data_path = './vector_store/data/SKALA전달용_College등록과정_v.0.1.xlsx'
    embed_model = os.getenv("EMBEDDING_MODEL_NAME")
    chromDB_path = os.getenv("VECTOR_DB_DIR")
    collection_name = os.getenv("LEC_COLLECTION_NAME")

    _college_raw = read_excel_file(data_path, 'Sheet1')

    columns_to_drop = ['사업별 교육체계', '공개여부']
    college_raw = _college_raw.drop(columns=columns_to_drop, axis=1)

    create_chromadb_local(college_raw, collection_name, embed_model, chromDB_path)
