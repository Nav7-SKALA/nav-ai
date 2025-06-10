import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import roleModel_prompt
from config import MODEL_NAME, TEMPERATURE
from vector_store.chroma_client import get_chroma_client
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import warnings

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
HISTORY_COLLECTION_NAME = os.getenv("HISTORY_COLLECTION_NAME")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")

# 경고 무시
warnings.filterwarnings("ignore", message="Add of existing embedding*")

# 1) ChromaDB 클라이언트
chroma_client = get_chroma_client(persist_directory=VECTOR_DB_DIR)

# 2) 컬렉션 얻기 (없으면 생성)
collection = chroma_client.get_or_create_collection(name=HISTORY_COLLECTION_NAME)

# 3) Vectorstore 래핑
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
employee_vectorstore = Chroma(
    client=chroma_client,
    collection_name=HISTORY_COLLECTION_NAME,
    embedding_function=embedding_model,
)
# 3) LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# 4) 프롬프트 체인
roleModel_chain = PromptTemplate.from_template(roleModel_prompt) | llm | StrOutputParser()

def roleModel_invoke(state: dict, config=None) -> dict:
    """roleModel Chain 실행 함수"""
    
    messages_text = "\n".join([
        msg.content for msg in state.get("messages", [])
        if hasattr(msg, 'content')
    ])
    
    # 1단계: 벡터DB에서 10명의 후보자 검색 (실제 유사도 점수 포함)
    related_people_with_scores = employee_vectorstore.similarity_search_with_score(messages_text, k=10)
    
    # LLM에게 보낼 정보 문자열 생성 (이 부분만 필수)
    employee_info = ""
    for i, (doc, score) in enumerate(related_people_with_scores, 1):
        profile_id = doc.metadata.get('profileId', 'unknown')
        employee_info += f"직원 {i} (profileId: {profile_id}, 유사도: {round(score, 3)}):\n{doc.page_content}\n\n"

    # LLM 호출
    result = roleModel_chain.invoke({
        "messages_text": messages_text,
        "employee_info": employee_info
    })

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