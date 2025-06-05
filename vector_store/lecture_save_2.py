import os
import re
import pandas as pd
import openpyxl
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_chroma import Chroma


# 환경변수 로드
load_dotenv()

def read_excel_file(file_path, sheet_name=0):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return None
    except Exception as e:
        print(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None


def build_skill_set_sentences(learning_guide):
    parts = re.split(r',(?=(?:특화|추천|공통필수)\s*-)', learning_guide)
    parsed = []
    for part in parts:
        key, domain, jobs_str = [x.strip() for x in part.split(" - ", maxsplit=2)]
        jobs_list = [job.strip() for job in jobs_str.split(",") if job.strip()]
        parsed.append({"key": key, "domain": domain, "jobs": jobs_list})

    skill_types = ['특화', '추천', '공통필수']
    grouped = {stype: defaultdict(list) for stype in skill_types}
    for item in parsed:
        k = item.get('key')
        if k in grouped:
            domain = item['domain']
            grouped[k][domain].extend(item['jobs'])

    lines = []
    for stype in skill_types:
        domain_dict = grouped.get(stype, {})
        if not domain_dict:
            continue
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
    return "\n".join(lines)


def trans_to_document(data_row):
    if pd.isna(data_row['Learning Guide']):
        skill_parse = ''
    else:
        skill_parse = build_skill_set_sentences(data_row['Learning Guide'])

    content_text = (
        f"**{data_row['교육과정명']}** 강의입니다. "
        f"이 과정은 {data_row['학부']} 학부의 {data_row['표준과정']} 표준 과정이며, "
        f"{data_row['교육유형']} 유형의 {data_row['학습유형']} 수업입니다. "
        f"총 학습 시간은 {data_row['학습시간']}시간입니다. "
        f"{skill_parse}"
    )
    return content_text


def create_chromadb_local(data, db_name, embed_model, persist_directory):
    """langchain-chroma + sentence_transformers 기반 벡터 저장"""

    model = SentenceTransformer(embed_model)
    documents = []
    embeddings = []

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

        documents.append(Document(page_content=content_text, metadata=metadata))
        embeddings.append(model.encode(content_text))

    # Chroma 저장
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=None,  # 직접 임베딩 주입하므로 None
        collection_name=db_name,
        persist_directory=persist_directory,
        embedding_function=lambda texts: embeddings
    )

    vectorstore.persist()
    print("✅ ChromaDB 저장 완료!")


if __name__ == "__main__":
    # 사내 강의 데이터 로컬 경로에 저장하는 초기화 코드

    data_path = './vector_store/data/SKALA전달용_College등록과정_v.0.1.xlsx'
    embed_model = os.getenv("EMBEDDING_MODEL_NAME")
    chromDB_path = os.getenv("VECTOR_DB_DIR")
    collection_name = os.getenv("LEC_COLLECTION_NAME")

    college_raw = read_excel_file(data_path, 'Sheet1')
    print(college_raw)

    # columns_to_drop = ['사업별 교육체계', '공개여부']
    # college_raw = college_raw.drop(columns=columns_to_drop, axis=1)

    create_chromadb_local(college_raw, collection_name, embed_model, chromDB_path)
