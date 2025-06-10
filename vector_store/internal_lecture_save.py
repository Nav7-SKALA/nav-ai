import os
import re
import pandas as pd
import openpyxl
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings 

## local version. (database setting X)

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
        

# 난이도 매핑
difficulty_map = {
    'Basic': '초급',
    'Intermediate': '중급',
    'Advanced': '고급',
    'Expert': '전문가'
}

# 스킬셋 document 문장 생성
def build_skill_set_sentences(learning_guide: str) -> list[str]:
    parts = re.split(r',(?=(?:특화|추천|공통필수)\s*-)', learning_guide)
    parsed = []
    for part in parts:
        key, domain, jobs_str = [x.strip() for x in part.split(" - ", maxsplit=2)]
        jobs_list = [job.strip() for job in jobs_str.split(",") if job.strip()]
        parsed.append({"key": key, "domain": domain, "jobs": jobs_list})

    skill_types = ['특화', '추천', '공통필수']
    grouped = {stype: defaultdict(list) for stype in skill_types}

    for item in parsed:
        k = item["key"]
        if k in grouped:
            grouped[k][item["domain"]].extend(item["jobs"])

    lines: list[str] = []
    for stype in skill_types:
        domain_dict = grouped.get(stype, {})
        if not domain_dict:
            continue

        pieces = []
        for domain, jobs in domain_dict.items():
            if not jobs:
                continue
            if len(jobs) == 1:
                pieces.append(f"{domain} 직무의 {jobs[0]}")
            else:
                pieces.append(f"{domain} 직무의 " + "와 ".join(jobs))
        joined = ", ".join(pieces)
        lines.append(f"{stype} skill set은 {joined}입니다.")

    return lines

# mysuni 강의 설명 생성
def build_mysuni_sentences(row: pd.Series) -> str:
    return (
        f"[{row['카테고리명']}] 카테고리의 '{row['채널명']}' 채널에서 제공하는 "
        f"{difficulty_map.get(row['난이도'], '정보없음')} 수준의 강의입니다. "
        f"{row['이수자수']}명이 수강하였고, 평균 평점은 {row['평점']}점입니다. "
        f"이 강의는 '{row['직무']}' 직무에 적합하며, "
        f"{row['Skill set']} 역량 향상에 도움을 줍니다."
    )

def trans_to_document(row):
    """입력된 한 행 데이터를 저장할 document text로 변환"""

    # 1) Skill Set 문장 생성(리스트 → "\n"으로 합치기)
    skill_parse = ""
    if pd.notna(row.get('Learning Guide')):
        skill_lines = build_skill_set_sentences(row['Learning Guide'])
        skill_parse = "\n".join(skill_lines)

    # 2) mySUNI 설명 생성
    mysuni_raw = ""
    if pd.notna(row.get('카테고리명')):
        mysuni_raw = build_mysuni_sentences(row)

    # 3) 문서 본문 텍스트 조합
    doc_text = (
        f"**{row['교육과정명']}** 강의입니다. "
        f"이 과정은 {row['학부']} 학부의 {row['표준과정']} 표준 과정이며, "
        f"{row['교육유형']} 유형의 {row['학습유형']} 수업입니다. "
        f"총 학습 시간은 {row['학습시간']}시간입니다.\n"
        f"{skill_parse}\n"
        f"{mysuni_raw}"
    )

    return doc_text


def embed_fn(texts: list[str]) -> list[list[float]]:
    """
    Chroma에 넘겨줄 임베딩 함수.
    texts: 문자열 목록을 받아서, 각 문장에 대해 SentenceTransformer로 임베딩 벡터 계산 후 리스트 반환.
    """
    return model.encode(texts).tolist()

def create_chromadb_local(data, embedding_function, persist_directory): # embed_fn 대신 embedding_function으로 변경
    """로컬 저장소(persist_directory)에 chromaDB 생성, 저장"""

    chroma = Chroma(
        collection_name=os.getenv("LEC_COLLECTION_NAME"),
        embedding_function=embedding_function, # embed_fn 대신 embedding_function으로 변경
        persist_directory=persist_directory
    )

    documents = []
    for i in tqdm(range(len(data))):
        row = data.iloc[i]

        content_text = trans_to_document(row)

        metadata = {
        "강의명": row['교육과정명'],
        "학부": row['학부'],
        "표준과정": row['표준과정'],
        "교육유형": row['교육유형'],
        "학습유형": row['학습유형'],
        "학습시간": row['학습시간'],
        "난이도": difficulty_map.get(row['난이도'], '정보없음')
        }
        documents.append(Document(page_content=content_text, metadata=metadata))

    chroma.add_documents(documents)



if __name__ == "__main__":

    college_path = os.path.join(os.getenv("DATA_DIR"), 'SKALA전달용_College등록과정_v.0.1.xlsx')
    mysuni_path = os.path.join(os.getenv("DATA_DIR"), 'SKALA전달용_mySUNI 과정리스트_v.0.1.xlsx')

    embed_model_name = os.getenv("EMBEDDING_MODEL_NAME") # 변수명 변경
    chromDB_path = os.getenv("VECTOR_DB_DIR")
    # model = SentenceTransformer(embed_model_name) # 이 줄은 더 이상 필요 없습니다.

    # HuggingFaceEmbeddings를 사용하여 SentenceTransformer 모델을 래핑합니다.
    # model_kwargs는 SentenceTransformer에 전달될 인자입니다.
    # encode_kwargs는 model.encode()에 전달될 인자입니다.
    embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    college_raw = read_excel_file(college_path, 'Sheet1')
    mysuni_raw = read_excel_file(mysuni_path)
    # merge
    merged_left = pd.merge(college_raw, mysuni_raw, 
                           left_on="교육과정명", right_on="카드명", 
                           how="left").drop(columns=['카드명', '인정학습시간(시간)'])

    create_chromadb_local(merged_left, embedding_function, chromDB_path) 
    
    print(f'{os.getenv("LEC_COLLECTION_NAME")}에 데이터 저장 완료!')
    