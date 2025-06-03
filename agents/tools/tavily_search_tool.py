from typing import Optional, List, Dict, Any
from tavily import TavilyClient
import os
from langchain.tools import tool
from dotenv import load_dotenv

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

@tool
def search_conferences(field: str, location: str = "Seoul") -> str:
    """특정 분야의 컨퍼런스를 검색합니다."""
    query = f"{field} conference 2025 in {location}"
    results = _tavily_search(query, "general", None, 3, True)
    return f"{field} 분야 {location} 컨퍼런스 정보:\n" + "\n".join(results)

@tool
def search_certifications(field: str) -> str:
    """특정 분야의 자격증을 검색합니다."""
    query = f"{field} certification recommended popular"
    results = _tavily_search(query, "general", None, 3, True)
    return f"{field} 분야 추천 자격증:\n" + "\n".join(results)
