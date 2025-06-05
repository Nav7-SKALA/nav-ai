import os
import sys
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import roleModel_prompt
from config import MODEL_NAME, TEMPERATURE, DB_PATH

from vector_store.employee_save import ensure_employee_vector_db_exists
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# 환경변수 로드
load_dotenv()


# 최초 실행시 백터DB 없으면 생성
ensure_employee_vector_db_exists()

# 벡터스토어 초기화
embedding_model = SentenceTransformerEmbeddings(model_name=VECTOR_DB_MODEL)
employee_vectorstore = Chroma(
    collection_name=HISTORY_COLLECTION,
    embedding_function=embedding_model,
    persist_directory=DB_PATH
)

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

roleModel_chain = PromptTemplate.from_template(roleModel_prompt) | llm | StrOutputParser()

def roleModel_invoke(state: dict, config=None) -> dict:
    """roleModel Chain 실행 함수"""
    
    messages_text = "\n".join([
        msg.content for msg in state.get("messages", [])
        if hasattr(msg, 'content')
    ])
    
    # 1단계: 벡터DB에서 9명의 후보자 검색 (실제 유사도 점수 포함)
    related_people_with_scores = employee_vectorstore.similarity_search_with_score(messages_text, k=9)
    
    # 점수별 매핑 딕셔너리 생성
    score_mapping = {}
    employee_info = ""
    
    for i, (doc, score) in enumerate(related_people_with_scores, 1):
        profile_id = doc.metadata.get('profileId', 'unknown')
        score_mapping[profile_id] = round(score, 3)  # 실제 유사도 점수 저장
        employee_info += f"직원 {i} (profileId: {profile_id}, 유사도: {round(score, 3)}):\n{doc.page_content}\n\n"
    
    # LLM으로 최종 3명 선택
    result = roleModel_chain.invoke({"query": roleModel_prompt})
    
    new_messages = list(state.get("messages", []))
    new_messages.append(AIMessage(content=result, name="RoleModel"))
    
    return {
        **state,
        "messages": new_messages
    }

def roleModel_node(state):
    """RoleModel 노드 함수"""
    return roleModel_invoke(state)

if __name__ == "__main__":
    # 테스트
    test_state = {"messages": [HumanMessage(content="3년차 PM 롤모델 추천해줘")]}
    result = roleModel_invoke(test_state)
    print(result["messages"][-1].content)