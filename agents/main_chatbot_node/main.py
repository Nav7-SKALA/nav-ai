import os
import asyncio
from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# 노드 함수들 import
from coursera_node import search_coursera_courses
from internal_lecture_node import lecture_search
from certifications_node import search_certifications
from config import MODEL_NAME, TEMPERATURE

# 환경변수 로드
load_dotenv()

# LLM 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# State 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    coursera_result: str
    internal_result: str
    certifications_result: str

# Supervisor prompt
supervisor_prompt = """You are a learning path recommendation system.

User Question: {user_question}
Key Keywords: [Extract technical terms in English from the question (e.g., "클라우드"→"cloud")]
IMPORTANT: First extract the key English keyword from the user question:

Search Results:
- Coursera External Courses: {coursera_result}
- Internal Company Courses: {internal_result}  
- Certification Information: {certifications_result}

Please organize and respond in the following clean format:

🎯 [Keyword] Learning Recommendation

📚 Recommended Courses
1. **[Course Name]** (Internal/Coursera)
   - Level: Beginner/Intermediate/Advanced
   - Description: [Brief description]

🏆 Related Certifications (if available)
- **[Certification Name]**: [Issuing Organization]

💡 Learning Path
[Simple guide in basic→advanced order]"""

supervisor_template = ChatPromptTemplate.from_template(supervisor_prompt)
supervisor_chain = supervisor_template | llm

# 병렬 처리를 위한 노드
async def parallel_search_node(state):
    """3개 검색을 병렬로 실행"""
    async def run_coursera():
        result = search_coursera_courses(state)
        return result["messages"][0].content
    
    async def run_internal():
        result = lecture_search(state)
        return result["messages"][0].content
    
    async def run_certifications():
        result = search_certifications(state)
        return result["messages"][0].content
    
    # 병렬 실행
    coursera_result, internal_result, cert_result = await asyncio.gather(
        run_coursera(),
        run_internal(), 
        run_certifications()
    )
    
    return {
        "coursera_result": coursera_result,
        "internal_result": internal_result,
        "certifications_result": cert_result
    }

def supervisor_node(state):
    """모든 검색 결과를 분석하여 최종 답변 생성"""
    user_question = state["messages"][0].content
    
    final_response = supervisor_chain.invoke({
        "user_question": user_question,
        "coursera_result": state.get("coursera_result", "검색 결과 없음"),
        "internal_result": state.get("internal_result", "검색 결과 없음"), 
        "certifications_result": state.get("certifications_result", "검색 결과 없음")
    })
    
    return {"messages": [AIMessage(content=final_response.content)]}

def create_graph():
    """그래프 생성 - 진짜 병렬 처리"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("parallel_search", parallel_search_node)
    workflow.add_node("supervisor", supervisor_node)
    
    # 순차 실행: START → parallel_search → supervisor → END
    workflow.add_edge(START, "parallel_search")
    workflow.add_edge("parallel_search", "supervisor")
    workflow.add_edge("supervisor", END)
    
    return workflow.compile()

if __name__ == "__main__":
    import time
    
    # 1. 각 노드 결과값 보여주는 테스트
    async def test_each_node():
        print("=== 각 노드별 결과 테스트 ===")
        query = "클라우드 어떻게 공부하지?"
        test_state = {
            "messages": [HumanMessage(content=query)],
            "coursera_result": "",
            "internal_result": "",
            "certifications_result": ""
        }
        
        start_time = time.time()
        result = await parallel_search_node(test_state)
        end_time = time.time()
        
        print(f"실행 시간: {end_time - start_time:.2f}초\n")
        print("📚 Coursera 결과:")
        print(result['coursera_result'])
        print("\n🏢 Internal 결과:")
        print(result['internal_result'])
        print("\n🏆 Certifications 결과:")
        print(result['certifications_result'])
        print("-" * 50)
        
        return result
    
    # 2. 최종 결과값만 보여주는 테스트
    async def test_final_result():
        print("\n=== 최종 결과 테스트 ===")
        query = "클라우드 어떻게 공부하지?"
        
        # 전체 그래프 실행
        graph = create_graph()
        final_result = await graph.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        
        print("🎯 최종 추천 결과:")
        print(final_result["messages"][-1].content)
    
    # 실행
    async def run_tests():
        # 각 노드 결과 확인
        # await test_each_node()
        
        # 최종 결과 확인
        await test_final_result()
    
    asyncio.run(run_tests())