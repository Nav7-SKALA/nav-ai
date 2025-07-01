import chromadb
import os
from sentence_transformers import SentenceTransformer
from chroma_client import get_chroma_client
from dotenv import load_dotenv

load_dotenv()

def create_embedding_text(career_step):
    """경력 단계를 임베딩용 텍스트로 변환"""
    text_parts = []
    
    for key, value in career_step.items():
        if key != "profileId" and value and str(value).strip():
            text_parts.append(f"{key}: {value}")
    
    return " | ".join(text_parts) 

def add_profile_to_vectordb(backend_data: dict) -> bool:
    """백엔드 데이터를 기존 형식으로 변환해서 VectorDB에 추가"""
    try:
        client = get_chroma_client()
        collection_name = os.getenv("JSON_HISTORY_COLLECTION_NAME")
        collection = client.get_collection(name=collection_name)
        
        # 데이터에서 profileId 추출
        user_info = backend_data.get("user_info", {})
        profile_id = str(user_info.get('profileId', ''))
        
        # 프로필 존재 여부 확인
        results = collection.get(where={"profileId": profile_id})
        if results['metadatas']:
            print(f"프로필 {profile_id} 이미 존재. 저장하지 않음")
            return False
        
        # 임베딩 모델 로드
        embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME"))
        
        documents = []
        metadatas = []
        ids = []
        
        projects = backend_data.get("projects", [])
        
        # 프로젝트가 없으면 기본 빈 데이터 1개 생성
        if not projects:
            projects = [{}]
        
        for i, project in enumerate(projects):
            career_step = {
                "연차": f"{project.get('startYear', '')}~{project.get('endYear', '')}년차" if project.get('startYear') and project.get('endYear') else "",
                "프로젝트규모": project.get('projectSize', ''),
                "역할": ', '.join(project.get('roles', [])) if project.get('roles') else "",
                "스킬셋": ', '.join(project.get('skillSets', [])) if project.get('skillSets') else "",
                "도메인": project.get('domainName', ''),
                "요약": f"{project.get('projectName', '')}. {project.get('projectDescribe', '')}" if project.get('projectName') or project.get('projectDescribe') else "",
                "터닝포인트": project.get('isTurningPoint', ''),
                "자격증": ', '.join([cert.get('name', '') for cert in backend_data.get('certifications', [])]) if backend_data.get('certifications') else "",
                "경험": ', '.join([exp.get('experienceName', '') for exp in backend_data.get('experiences', [])]) if backend_data.get('experiences') else "",
                "총경력년수": user_info.get('years', '')
            }
            
            # 임베딩용 텍스트 생성
            embedding_text = create_embedding_text(career_step)
            documents.append(embedding_text)
            
            # 메타데이터 생성
            metadata = {
                "profileId": profile_id,
                "연차": career_step["연차"],
                "프로젝트규모": career_step["프로젝트규모"],
                "역할": career_step["역할"],
                "스킬셋": career_step["스킬셋"],
                "도메인": career_step["도메인"],
                "요약": career_step["요약"],
                "터닝포인트": career_step["터닝포인트"],
                "자격증": career_step["자격증"],
                "경험": career_step["경험"],
                "총경력년수": career_step["총경력년수"]
            }
            metadatas.append(metadata)
            ids.append(f"{profile_id}_{i}")
        
        # 임베딩 생성
        embeddings_list = embedding_model.encode(documents).tolist()
        
        # ChromaDB에 추가
        collection.add(
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"프로필 {profile_id} 추가 완료 ({len(documents)}개 프로젝트)")
        return True
        
    except Exception as e:
        print(f"프로필 추가 오류: {e}")
        return False