import pandas as pd
import json
import random
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chroma_client import get_chroma_client


# 환경 변수 로드
load_dotenv()

def assign_grade(emp_id):
    """사번에 따라 grade 할당"""
    cl4_emp_ids = [
        "EMP-525170", "EMP-366273", "EMP-343203", "EMP-253717", "EMP-407037",
        "EMP-408705", "EMP-408778", "EMP-408851", "EMP-408924", "EMP-409028",
        "EMP-409101", "EMP-409174", "EMP-409320", "EMP-409393", "EMP-409466",
        "EMP-409539", "EMP-409612", "EMP-409685", "EMP-409789", "EMP-409862",
        "EMP-409935", "EMP-410008", "EMP-410081", "EMP-410154", "EMP-410227",
        "EMP-410300", "EMP-410373", "EMP-410446", "EMP-410623", "EMP-410696",
        "EMP-410769", "EMP-576352", "EMP-714980", "EMP-577655", "EMP-347988",
        "EMP-262679", "EMP-202827", "EMP-198261"
    ]
    
    return "CL4" if emp_id in cl4_emp_ids else random.choice(["CL1", "CL2", "CL3"])

def determine_project_scale(scale_num):
    """프로젝트 규모 숫자를 텍스트로 변환"""
    if pd.isna(scale_num):
        return ""
    
    scale_map = {
        1: "소형",
        2: "중형", 
        3: "대형",
        4: "초대형"
    }
    return scale_map.get(int(scale_num), "")

def create_embedding_text(career_step):
    """경력 단계를 임베딩용 텍스트로 변환"""
    text_parts = []
    
    for key, value in career_step.items():
        if key != "profileId" and value and str(value).strip():
            text_parts.append(f"{key}: {value}")
    
    return " | ".join(text_parts)

def excel_to_chroma(excel_file):
    """엑셀 파일을 처리하여 ChromaDB에 저장"""
    
    # 환경 변수 가져오기
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    JSON_HISTORY_COLLECTION_NAME = os.getenv("JSON_HISTORY_COLLECTION_NAME")

        
    # 임베딩 모델 생성
    embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 엑셀 파일 읽기
    df = pd.read_excel(excel_file)
    
    employees = {}
    
    for _, row in df.iterrows():
        emp_id = str(row['ID']).strip()

        if pd.isna(row['StartYear']) or pd.isna(row['EndYear']):
            continue
        
        # 스킬셋 수집
        skillsets = []
        for i in range(1, 5):
            skill_col = f'SkillSet{i}'
            if skill_col in row and pd.notna(row[skill_col]) and str(row[skill_col]).strip():
                skill = str(row[skill_col]).strip()
                if skill not in skillsets:
                    skillsets.append(skill)
        
        # 요약 생성
        summary_parts = []
        if pd.notna(row['Project']) and str(row['Project']).strip():
            summary_parts.append(str(row['Project']).strip())
        
        # 커리어 영향 (1인 경우만)
        if (pd.notna(row['CareerImpact']) and int(row['CareerImpact']) == 1 and
            pd.notna(row['CareerImpactDesc']) and str(row['CareerImpactDesc']).strip()):
            summary_parts.append(str(row['CareerImpactDesc']).strip())
        
        summary = '. '.join(summary_parts)
        
        # 경력 단계 생성
        career_step = {
            "연차": f"{int(row['StartYear'])}~{int(row['EndYear'])}년차",
            "프로젝트규모": determine_project_scale(row['ProjectScale']),
            "역할": str(row['Roles']).strip() if pd.notna(row['Roles']) else "",
            "스킬셋": ', '.join(skillsets),
            "도메인": str(row['Industry']).strip() if pd.notna(row['Industry']) else "",
            "요약": summary
        }
        
        # 사원별로 그룹화
        if emp_id not in employees:
            employees[emp_id] = {
                "사번": emp_id,
                "입사년도": random.randint(2005, 2020),
                "grade": assign_grade(emp_id),
                "경력흐름": []
            }
        
        employees[emp_id]["경력흐름"].append(career_step)
    
    # 중복 제거
    for emp_data in employees.values():
        unique_careers = []
        seen_projects = set()
        
        for career in emp_data["경력흐름"]:
            project_key = (
                career["연차"], career["프로젝트규모"], 
                career["역할"], career["도메인"], career["요약"]
            )
            
            if project_key not in seen_projects:
                unique_careers.append(career)
                seen_projects.add(project_key)
            else:
                # 중복된 경우 스킬셋 병합
                for existing_career in unique_careers:
                    existing_key = (
                        existing_career["연차"], existing_career["프로젝트규모"],
                        existing_career["역할"], existing_career["도메인"], existing_career["요약"]
                    )
                    if existing_key == project_key:
                        existing_skills = set(existing_career["스킬셋"].split(', ')) if existing_career["스킬셋"] else set()
                        new_skills = set(career["스킬셋"].split(', ')) if career["스킬셋"] else set()
                        merged_skills = existing_skills.union(new_skills)
                        existing_career["스킬셋"] = ', '.join(sorted(merged_skills)) if merged_skills else ""
                        break
        
        emp_data["경력흐름"] = unique_careers
    
    # JSON 저장
    employees_list = list(employees.values())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_output_file = os.path.join(script_dir, "employees_data.json")
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(employees_list, f, ensure_ascii=False, indent=2)
    print(f"JSON 파일 저장 완료: {json_output_file}")
    
    # ChromaDB에 저장
    print("ChromaDB에 데이터 저장 중...")
    
    # ChromaDB 클라이언트 및 컬렉션 생성
    chroma_client = get_chroma_client()

    # 기존 컬렉션 삭제 후 새로 생성
    try:
        chroma_client.delete_collection(name=JSON_HISTORY_COLLECTION_NAME)
        print(f"기존 컬렉션 '{JSON_HISTORY_COLLECTION_NAME}' 삭제됨")
    except:
        print(f"삭제할 컬렉션 '{JSON_HISTORY_COLLECTION_NAME}'이 없음")

    # 새 컬렉션 생성
    collection = chroma_client.create_collection(name=JSON_HISTORY_COLLECTION_NAME)
    print(f"새 컬렉션 '{JSON_HISTORY_COLLECTION_NAME}' 생성")

    # 사번별로 profileId 매핑 생성
    emp_id_to_profile_id = {}
    profile_id_counter = 1

    for emp_data in employees.values():
        emp_id = emp_data["사번"]
        if emp_id not in emp_id_to_profile_id:
            emp_id_to_profile_id[emp_id] = str(profile_id_counter)
            profile_id_counter += 1

    # 문서, 메타데이터, ID 준비
    documents = []
    metadatas = []
    ids = []

    for emp_data in employees.values():
        emp_id = emp_data["사번"]
        profile_id = emp_id_to_profile_id[emp_id]
        
        for i, career in enumerate(emp_data["경력흐름"]):
            # 임베딩용 텍스트 생성
            embedding_text = create_embedding_text(career)
            documents.append(embedding_text)
            
            # 메타데이터 생성
            metadata = {
                "profileId": profile_id,
                "사번": emp_data["사번"],
                "입사년도": emp_data["입사년도"],
                "grade": emp_data["grade"],
                "연차": career["연차"],
                "프로젝트규모": career["프로젝트규모"],
                "역할": career["역할"],
                "스킬셋": career["스킬셋"],
                "도메인": career["도메인"],
                "요약": career["요약"]
            }
            metadatas.append(metadata)
            
            # 고유 ID 생성
            unique_id = f"{profile_id}_{i}"
            ids.append(unique_id)

    # 임베딩 생성
    print("임베딩 생성 중...")
    embeddings_list = embeddings.encode(documents).tolist()

    # 배치 크기 설정 (50개씩 나누어 전송)
    batch_size = 50
    total_docs = len(documents)

    print(f"총 {total_docs}개 문서를 {batch_size}개씩 나누어 전송...")

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch_docs = documents[i:end_idx]
        batch_embeddings = embeddings_list[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        print(f"배치 {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} 전송 중...")
        
        # ChromaDB에 배치 추가
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    print(f"ChromaDB 저장 완료: {total_docs}개의 경력 데이터가 저장되었습니다.")
    return total_docs

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(script_dir, "processed_output.xlsx")
    
    if not os.path.exists(excel_file):
        print("processed_output.xlsx 파일을 찾을 수 없습니다.")
        exit(1)
    
    doc_count = excel_to_chroma(excel_file)
    print(f"\n총 {doc_count}개의 문서가 ChromaDB에 저장되었습니다.")