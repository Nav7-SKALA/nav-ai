from typing import Optional, List, Dict, Any
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

def _tavily_search(query: str, topic: str = "general", start_date: Optional[str] = None,
                  max_results: int = 5, format_output: bool = False, 
                  api_key: Optional[str] = None) -> List[str]:
   """Tavily를 사용한 공통 검색 함수"""
   try:
       api_key = api_key or os.getenv("TAVILY_API_KEY")
       if not api_key:
           raise ValueError("TAVILY_API_KEY가 필요합니다.")
       
       client = TavilyClient(api_key=api_key)
       
       if start_date:
           query += f" after:{start_date}-01"
       
       search_params = {
           "query": query,
           "search_depth": "basic",
           "topic": topic,
           "max_results": max_results,
           "include_answer": True,
           "include_raw_content": False
       }
       
       response = client.search(**search_params)
       results = []
       
       if response.get("answer"):
           results.append(f"요약: {response['answer']}")
       
       for result in response.get("results", []):
           if format_output:
               formatted_result = f"제목: {result.get('title', 'N/A')}\n"
               formatted_result += f"URL: {result.get('url', 'N/A')}\n"
               formatted_result += f"내용: {result.get('content', 'N/A')}\n"
               if result.get('published_date'):
                   formatted_result += f"게시일: {result.get('published_date')}\n"
               formatted_result += "-" * 50
               results.append(formatted_result)
           else:
               results.append(result.get('content', ''))
       
       return results
       
   except Exception as e:
       return [f"검색 중 오류 발생: {str(e)}"]

def search_certifications(state):
   """특정 분야의 자격증을 검색합니다."""
   try:
       # state에서 사용자 질문 추출
       messages = state["messages"]
       user_query = messages[0].content
       
       # 사용자 질문에서 분야 추출 (간단한 방법)
       field = user_query  # 또는 더 정교한 파싱 로직 추가
       
       query = f"{field} certification recommended popular"
       results = _tavily_search(query, "general", None, 2, True)
       
       content = f"{field} 분야 추천 자격증:\n" + "\n".join(results)
       
       return {"messages": [AIMessage(content=content)]}
       
   except Exception as e:
       error_message = f"자격증 검색 중 오류 발생: {str(e)}"
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
       "messages": [HumanMessage(content="AI 관련 자격증 추천")]
   }
   
   # 함수 실행
   result = search_certifications(test_state)
   
   # 결과 출력
   print("검색 결과:")
   print(result["messages"][0].content)