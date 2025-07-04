import os
from typing import Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import numpy as np
from vector_store.chroma_client import get_chroma_client
from agents.main_chatbot.config import MODEL_NAME, TEMPERATURE
from agents.main_chatbot.prompt import lecture_prompt
from agents.main_chatbot.response import LectureRecommendation

load_dotenv()


class EmbeddingModel:
    def __init__(self):
        """한국어 임베딩 모델 초기화"""
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME")
        self.model = SentenceTransformer(self.model_name)
    
    def embed_query(self, query):
        """쿼리 텍스트를 임베딩 벡터로 변환"""
        return self.model.encode(query).tolist()

def lecture_search(query: str) -> list:
    """vectorDB에서 정보 검색 도구"""
    try:
        print("DEBUG확인하자 9:35분")
        print(f"DEBUG - lecture_search query: '{query}'")
        print(f"DEBUG - LEC_COLLECTION_NAME: {os.getenv('LEC_COLLECTION_NAME')}")
        
        client = get_chroma_client()
        print(f"DEBUG - ChromaDB client created: {client}")
        
        collection_name = os.getenv("LEC_COLLECTION_NAME")
        collection = client.get_or_create_collection(collection_name)
        print(f"DEBUG - Collection: {collection}")
        print(f"DEBUG - Collection count: {collection.count()}")
        
        embed = EmbeddingModel()
        query_embedding = embed.embed_query(query) 
        print(f"DEBUG - Query embedding length: {len(query_embedding) if query_embedding else 0}")

        TOP_N = 5
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_N,
            include=["documents", "metadatas"]
        )
        
        print(f"DEBUG - ChromaDB results: {results}")
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        print(f"DEBUG - Documents count: {len(documents)}")
        print(f"DEBUG - Metadatas count: {len(metadatas)}")
        
        lectures_list = []
        for num, (doc, meta) in enumerate(zip(documents, metadatas)):
            lectures_list.append(f"강의{num+1}: {doc}, 메타데이터: {meta}")

        print(f"DEBUG - Final lectures_list: {lectures_list}")
        return lectures_list
        
    except Exception as e:
        print(f"DEBUG - lecture_search exception: {str(e)}")
        return []  # 빈 리스트 반환



# async def lecture_recommend(user_query: str, career_summary: str = "") -> Dict:
#     """사내 교육 추천 Tool 함수"""
#     try:
#         llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        
#         # 강의 검색
#         lecture_results = lecture_search(user_query)    

#         # print("이건 강의정보 확인!!!!")
#         # print(lecture_results)
#         # print("*"*60)
        
#         # 프롬프트 템플릿
#         lecture_recommend_template = PromptTemplate(
#             input_variables=["user_query", "available_courses", "career_summary"],
#             template=lecture_prompt
#         )
        
#         # LLM 실행
#         lecture_llm_chain = lecture_recommend_template | llm.with_structured_output(LectureRecommendation)
#         result = lecture_llm_chain.invoke({
#             "user_query": user_query,
#             "available_courses": lecture_results,
#             "career_summary": career_summary
#         })

#         # selected_lecture_name = result.internal_course
#         # print(""*60)
#         # print("이건 최종적으로 뽑힌 강의 정보 확인!!!")
#         # print(selected_lecture_name)
#         # print("*"*60)

#         return {
#             'internal_course': result.internal_course,
#             'ax_college': result.ax_college,
#             'explanation': result.explanation
#         }
        
#     except Exception as e:
#         return {
#             'internal_course': f"사내 교육 추천 중 오류 발생: {str(e)}",
#             'ax_college': "",
#             'explanation': ""
#         }

async def lecture_recommend(user_query: str, career_summary: str = "") -> Dict:
    """사내 교육 추천 Tool 함수"""
    try:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        
        # 강의 검색
        lecture_results = lecture_search(user_query)    
        print(f"DEBUG - lecture_results 타입: {type(lecture_results)}")
        print(f"DEBUG - lecture_results 길이: {len(lecture_results)}")
        
        # 프롬프트 템플릿
        lecture_recommend_template = PromptTemplate(
            input_variables=["user_query", "available_courses", "career_summary"],
            template=lecture_prompt
        )
        
        # LLM 실행
        lecture_llm_chain = lecture_recommend_template | llm.with_structured_output(LectureRecommendation)
        
        print(f"DEBUG - About to invoke LLM with structured output")
        result = lecture_llm_chain.invoke({
            "user_query": user_query,
            "available_courses": lecture_results,
            "career_summary": career_summary
        })

        return {
            'internal_course': result.internal_course,
            'ax_college': result.ax_college,
            'explanation': result.explanation
        }
        
    except Exception as e:
        print(f"DEBUG - DETAILED Exception: {str(e)}")
        print(f"DEBUG - Exception type: {type(e)}")
        import traceback
        print(f"DEBUG - Full traceback: {traceback.format_exc()}")
        return {
            'internal_course': f"사내 교육 추천 중 오류 발생: {str(e)}",
            'ax_college': "",
            'explanation': ""
        }