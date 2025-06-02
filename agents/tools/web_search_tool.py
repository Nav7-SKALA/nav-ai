from langchain.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

### test
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool = TavilySearchResults(max_results=5)


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """주어진 쿼리에 대해 웹 검색을 수행하고 관련 정보를 반환합니다.

    Args:
        query (str): 검색할 쿼리 문자열.

    Returns:
        str: 웹 검색 결과 요약 또는 관련 정보.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")

    tavily = TavilyClient(api_key=tavily_api_key)
    
    # Tavily API를 사용하여 검색 수행
    # 'search_depth': 'basic' 또는 'advanced' (더 깊은 검색을 원하면 'advanced')
    # 'include_answer': True (검색 결과에서 직접 답변을 추출)
    # 'include_raw_content': False (원시 HTML 콘텐츠는 포함하지 않음)
    response = tavily.search(query=query, search_depth='basic', include_answer=True)
    
    # 검색 결과에서 답변 또는 요약된 내용 반환
    if response.get("answer"):
        return response["answer"]
    elif response.get("results"):
        # 여러 검색 결과가 있을 경우, 각 결과의 내용을 조합하여 반환
        combined_results = []
        for res in response["results"]:
            combined_results.append(f"Title: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}\n")
        return "\n---\n".join(combined_results)
    else:
        return "검색 결과를 찾을 수 없습니다."


### daeun

import requests
import json
import re
import feedparser
import urllib.parse
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus

def _extract_courses(html: str, limit: int) -> List[Dict[str, Any]]:
    """HTML에서 강의 정보 추출"""
    courses = []
    course_links = re.findall(r'href="(/learn/[^"]+)"', html)
    
    # 중복 제거
    unique_links = []
    seen_slugs = set()
    for link in course_links:
        slug = link.split('?')[0]
        if slug not in seen_slugs:
            seen_slugs.add(slug)
            unique_links.append(slug)
    
    for link in unique_links[:limit]:
        try:
            course_slug = link.replace('/learn/', '')
            course_name = course_slug.replace('-', ' ').title()
            
            # 제목 추출 패턴
            broader_patterns = [
                rf'href="{re.escape(link)}[^"]*"[^>]*>([^<]+)<',
                rf'href="{re.escape(link)}[^"]*"[^>]*>[^<]*<[^>]*>([^<]+)<',
                rf'href="{re.escape(link)}[^"]*"[^>]*>[^<]*<[^>]*>[^<]*<[^>]*>([^<]+)<'
            ]
            
            for pattern in broader_patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                    if title and len(title) > 3 and not title.isdigit():
                        course_name = title
                        break
            
            courses.append({
                'name': course_name,
                'url': f"https://www.coursera.org{link}",
            })
            
        except Exception:
            continue
    
    return courses

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
def search_coursera_courses(query: str, limit: int = 5) -> str:
    """Coursera에서 특정 주제의 강의를 검색합니다."""
    try:
        encoded_query = quote_plus(query)
        url = f"https://www.coursera.org/search?query={encoded_query}&entityTypeDescription=Courses"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        courses = _extract_courses(response.text, limit)
        
        if not courses:
            return f"'{query}' 관련 Coursera 강의를 찾을 수 없습니다."
        
        result = f"'{query}' 관련 Coursera 강의 {len(courses)}개:\n"
        for i, course in enumerate(courses, 1):
            result += f"{i}. {course['name']}\n   - {course['url']}\n"
        
        return result
        
    except Exception as e:
        return f"Coursera 검색 중 오류 발생: {str(e)}"

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


# 사용 예시 (Agent에서 호출될 때)
# if __name__ == "__main__":
#     # .env 파일에 TAVILY_API_KEY=YOUR_TAVILY_API_KEY 추가 필요
#     result = web_search("2024년 최신 AI 컨퍼런스")
#     print(result)