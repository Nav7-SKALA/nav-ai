import os
from typing import Optional, List, Dict
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
# import requests
# import json
# import re
# from typing import List, Dict, Any
# from urllib.parse import quote_plus

# def search_coursera_courses(query: str, limit: int = 10) -> str:
#     """
#     Coursera에서 특정 주제의 강의를 검색합니다.
    
#     Args:
#         query (str): 검색할 강의 주제
#         limit (int): 반환할 강의 개수
        
#     Returns:
#         str: JSON 형태의 강의 정보
#     """
#     try:
#         # 검색 URL 생성
#         encoded_query = quote_plus(query)
#         url = f"https://www.coursera.org/search?query={encoded_query}&entityTypeDescription=Courses"
        
#         # 기본 헤더
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
#         }
        
#         # 웹페이지 요청
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
        
#         # 강의 정보 추출
#         courses = extract_courses(response.text, limit)
        
#         if not courses:
#             return json.dumps({
#                 "success": False,
#                 "message": f"'{query}' 관련 강의를 찾을 수 없습니다.",
#                 "courses": []
#             }, ensure_ascii=False, indent=2)
        
#         return json.dumps({
#             "success": True,
#             "message": f"'{query}' 관련 강의 {len(courses)}개를 찾았습니다.",
#             "courses": courses
#         }, ensure_ascii=False, indent=2)
        
#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "message": f"오류 발생: {str(e)}",
#             "courses": []
#         }, ensure_ascii=False, indent=2)

# def extract_courses(html: str, limit: int) -> List[Dict[str, Any]]:
#     """HTML에서 강의 정보 추출"""
#     courses = []
    
#     # 강의 링크 찾기
#     course_links = re.findall(r'href="(/learn/[^"]+)"', html)
#     print(f"총 {len(course_links)}개 링크 발견")
    
#     # 중복 제거를 먼저 수행
#     unique_links = []
#     seen_slugs = set()
#     for link in course_links:
#         slug = link.split('?')[0]  # 쿼리 파라미터 제거
#         if slug not in seen_slugs:
#             seen_slugs.add(slug)
#             unique_links.append(slug)
    
#     print(f"중복 제거 후 {len(unique_links)}개 링크")
    
#     # 필요한 개수만큼만 처리
#     for link in unique_links[:limit]:
#         try:
#             # 링크에서 강의명 추출
#             course_slug = link.replace('/learn/', '')
#             course_name = course_slug.replace('-', ' ').title()
            
#             # 더 넓은 범위에서 제목 찾기
#             broader_patterns = [
#                 rf'href="{re.escape(link)}[^"]*"[^>]*>([^<]+)<',
#                 rf'href="{re.escape(link)}[^"]*"[^>]*>[^<]*<[^>]*>([^<]+)<',
#                 rf'href="{re.escape(link)}[^"]*"[^>]*>[^<]*<[^>]*>[^<]*<[^>]*>([^<]+)<'
#             ]
            
#             actual_title = None
#             for pattern in broader_patterns:
#                 match = re.search(pattern, html, re.IGNORECASE)
#                 if match:
#                     title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
#                     if title and len(title) > 3 and not title.isdigit():
#                         actual_title = title
#                         break
            
#             if actual_title:
#                 course_name = actual_title
            
#             courses.append({
#                 'name': course_name,
#                 'url': f"https://www.coursera.org{link}",
#             })
            
#         except Exception as e:
#             print(f"링크 처리 중 오류 {link}: {e}")
#             continue
    
#     print(f"최종 {len(courses)}개 강의 반환")
#     return courses

# # Tool definition
# COURSERA_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "search_coursera_courses",
#         "description": "Coursera에서 특정 주제와 관련된 온라인 강의를 검색합니다.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "검색할 강의 주제"
#                 },
#                 "limit": {
#                     "type": "integer",
#                     "description": "반환할 강의 개수 (기본값: 10)",
#                     "default": 10
#                 }
#             },
#             "required": ["query"]
#         }
#     }
# }

# # 테스트
# if __name__ == "__main__":
#     result = search_coursera_courses("AI", 1)
#     print(result)

import os
from typing import Optional, List
from tavily import TavilyClient

def _tavily_search(
    query: str,
    topic: str = "general",
    start_date: Optional[str] = None,
    max_results: int = 5,
    format_output: bool = False,
    api_key: Optional[str] = None
) -> List[str]:
    """공통 검색 함수"""
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

def search_conferences(
    field: str,
    location: str,
    start_date: Optional[str] = None,
    max_results: int = 5,
    format_output: bool = False,
    api_key: Optional[str] = None
) -> List[str]:
    """컨퍼런스 검색"""
    query = f"{field} conference 2025 in {location}"
    return _tavily_search(query, "general", start_date, max_results, format_output, api_key)

def search_certifications(
    field: str,
    start_date: Optional[str] = None,
    max_results: int = 5,
    format_output: bool = False,
    api_key: Optional[str] = None
) -> List[str]:
    """자격증 검색"""
    query = f"{field} certification recommended"
    return _tavily_search(query, "general", start_date, max_results, format_output, api_key)

def search_learning_platforms(
    field: str,
    start_date: Optional[str] = None,
    max_results: int = 5,
    format_output: bool = False,
    api_key: Optional[str] = None
) -> List[str]:
    """학습 플랫폼 검색"""
    query = f"{field} learning platform online course"
    return _tavily_search(query, "general", start_date, max_results, format_output, api_key)

def search_latest_tech(
    field: str,
    start_date: Optional[str] = None,
    max_results: int = 5,
    format_output: bool = False,
    api_key: Optional[str] = None
) -> List[str]:
    """최신 기술 검색"""
    query = f"{field} latest technology trends 2025"
    return _tavily_search(query, "news", start_date, max_results, format_output, api_key)

if __name__ == "__main__":
    # # 컨퍼런스 검색
    # conferences = search_conferences("AI", "Seoul", "2025-05", 2, True)
    # print("=== 컨퍼런스 ===")
    # print("\n".join(conferences))
    
    # # 자격증 검색
    # certs = search_certifications("cloud", "2020-01", 2, True)
    # print("\n=== 자격증 ===")
    # print("\n".join(certs))
    
    # # 학습 플랫폼 검색
    # platforms = search_learning_platforms("Python", "2020-01", 2, True)
    # print("\n=== 학습 플랫폼 ===")
    # print("\n".join(platforms))
    
    # 최신 기술 검색
    tech = search_latest_tech("mcp", "2025-01", 2, True)
    print("\n=== 최신 기술 ===")
    print("\n".join(tech))












# def tavily_search(
#     query: str,
#     topic: str = "general",
#     days: Optional[int] = None,
#     max_results: int = 5,
#     format_output: bool = False,
#     api_key: Optional[str] = None
# ) -> List[str]:

#     try:
#         # API 키 설정
#         api_key = api_key or os.getenv("TAVILY_API_KEY")
#         if not api_key:
#             raise ValueError("TAVILY_API_KEY가 필요합니다.")
        
#         # 클라이언트 생성
#         client = TavilyClient(api_key=api_key)

        
#         # 검색 파라미터 설정
#         search_params = {
#             "query": query,
#             "search_depth": "basic",
#             "topic": topic,
#             "max_results": max_results,
#             "include_answer": True,
#             "include_raw_content": False
#         }
        
#         # 날짜 제한이 있으면 추가
#         if days:
#             search_params["days"] = days
        
#         # 검색 실행
#         response = client.search(**search_params)
        
#         # 결과 처리
#         results = []
        
#         # 답변이 있으면 먼저 추가
#         if response.get("answer"):
#             results.append(f"요약: {response['answer']}")
        
#         # 검색 결과 처리
#         for result in response.get("results", []):
#             if format_output:
#                 formatted_result = f"제목: {result.get('title', 'N/A')}\n"
#                 formatted_result += f"URL: {result.get('url', 'N/A')}\n"
#                 formatted_result += f"내용: {result.get('content', 'N/A')}\n"
#                 formatted_result += "-" * 50
#                 results.append(formatted_result)
#             else:
#                 results.append(result.get('content', ''))
        
#         return results
        
#     except Exception as e:
#         return [f"검색 중 오류 발생: {str(e)}"]


# if __name__ == "__main__":
#     # 3가지 카테고리별 검색 파라미터 정의
#     categories: Dict[str, Dict] = {
#         # "자격증": {
#         #     "query": "cloud 관련 추천 자격증 목록",
#         #     "topic": "general",
#         #     "days": 30,           # 최근 30일간
#         #     "max_results": 2,
#         #     "format_output": True
#         # },
#         # "최신 기술": {
#         #     "query": "cloud 관련 최신 기술 트렌드",
#         #     "topic": "news",
#         #     "days": 7,            # 최근 일주일간
#         #     "max_results": 2,
#         #     "format_output": True
#         # },
#         "컨퍼런스": {
#             "query": "cloud 관련 추천 컨퍼런스",
#             "topic": "general",
#             "days": 30,           # 최근 두 달간
#             "max_results": 2,
#             "format_output": True
#         }
#     }
    
#     for cat_name, params in categories.items():
#         print(f"\n=== {cat_name} ===")
#         results = tavily_search(**params)
#         print("\n".join(results))