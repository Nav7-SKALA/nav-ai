import os
from dotenv import load_dotenv
import warnings

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage

from config import BASE_DIR
import sys
sys.path.append(BASE_DIR)
from vector_store.chroma_client import get_chroma_client


# 허깅페이스 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# .env 파일 로드
load_dotenv()

LEC_COLLECTION_NAME = os.getenv("LEC_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

if not LEC_COLLECTION_NAME or not EMBEDDING_MODEL_NAME:
    print("환경 변수 (LEC_COLLECTION_NAME, EMBEDDING_MODEL_NAME)가 제대로 설정되지 않았습니다.")
    exit()

def lecture_search(state):
    """vectorDB에서 정보 검색 도구"""
    try:
        # state에서 사용자 질문 추출
        messages = state["messages"]
        query = messages[0].content
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
        )

        # 원격 ChromaDB 클라이언트 생성
        chroma_client = get_chroma_client()
        
        # LangChain Chroma로 원격 연결
        vectordb = Chroma(
            client=chroma_client,
            collection_name=LEC_COLLECTION_NAME,  # HISTORY_COLLECTION_NAME -> LEC_COLLECTION_NAME 수정
            embedding_function=embeddings,
        )
        
        results = vectordb.similarity_search(query, k=2)
        
        # 결과 포맷팅
        if not results:
            content = f"'{query}' 관련 강의를 찾을 수 없습니다."
        else:
            content = f"'{query}' 관련 강의 {len(results)}개를 찾았습니다:\n\n"
            for i, doc in enumerate(results, 1):
                content += f"{i}. {doc.page_content}\n"
                if doc.metadata:
                    content += f"   메타데이터: {doc.metadata}\n"
                content += "\n"
        
        return {"messages": [AIMessage(content=content)]}
        
    except Exception as e:
        error_message = f"강의 검색 중 오류 발생: {str(e)}"
        return {"messages": [AIMessage(content=error_message)]}

if __name__ == "__main__":
    # 테스트용 import 추가
    from typing import Annotated, Sequence, TypedDict
    from langchain_core.messages import BaseMessage, HumanMessage
    from langgraph.graph.message import add_messages
    
    # State 정의
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 테스트용 state 생성
    test_state = {
        "messages": [HumanMessage(content="파이썬 기초 강의")]
    }
    
    # 함수 실행
    result = lecture_search(test_state)
    
    # 결과 출력
    print("검색 결과:")
    print(result["messages"][0].content)

