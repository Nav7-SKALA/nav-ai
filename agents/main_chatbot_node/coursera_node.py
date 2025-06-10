# import re
# from typing import Optional, List, Dict, Any
# from urllib.parse import quote_plus
# import requests
# from langchain.tools import tool

# def _extract_courses(html: str, limit: int) -> List[Dict[str, Any]]:
#     """HTML에서 강의 정보 추출"""
#     courses = []
#     course_links = re.findall(r'href="(/learn/[^"]+)"', html)
    
#     # 중복 제거
#     unique_links = []
#     seen_slugs = set()
#     for link in course_links:
#         slug = link.split('?')[0]
#         if slug not in seen_slugs:
#             seen_slugs.add(slug)
#             unique_links.append(slug)
    
#     for link in unique_links[:limit]:
#         try:
#             course_slug = link.replace('/learn/', '')
#             course_name = course_slug.replace('-', ' ').title()
            
#             # 제목 추출 패턴
#             broader_patterns = [
#                 rf'href="{re.escape(link)}[^"]*"[^>]*>([^<]+)<',
#                 rf'href="{re.escape(link)}[^"]*"[^>]*>[^<]*<[^>]*>([^<]+)<',
#                 rf'href="{re.escape(link)}[^"]*"[^>]*>[^<]*<[^>]*>[^<]*<[^>]*>([^<]+)<'
#             ]
            
#             for pattern in broader_patterns:
#                 match = re.search(pattern, html, re.IGNORECASE)
#                 if match:
#                     title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
#                     if title and len(title) > 3 and not title.isdigit():
#                         course_name = title
#                         break
            
#             courses.append({
#                 'name': course_name,
#                 'url': f"https://www.coursera.org{link}",
#             })
            
#         except Exception:
#             continue
    
#     return courses


# def search_coursera_courses(query: str, limit: int = 2) -> str:
#     """Coursera에서 특정 주제의 강의를 검색합니다."""
#     try:
#         encoded_query = quote_plus(query)
#         url = f"https://www.coursera.org/search?query={encoded_query}&entityTypeDescription=Courses"
        
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
#         }
        
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
        
#         courses = _extract_courses(response.text, limit)
        
#         if not courses:
#             return f"'{query}' 관련 Coursera 강의를 찾을 수 없습니다."
        
#         result = f"'{query}' 관련 Coursera 강의 {len(courses)}개:\n"
#         for i, course in enumerate(courses, 1):
#             result += f"{i}. {course['name']}\n   - {course['url']}\n"
        
#         return {"search_coursera": results}
        
#     except Exception as e:
#         return f"Coursera 검색 중 오류 발생: {str(e)}"



import re
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus
import requests
from langchain_core.messages import AIMessage

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

def search_coursera_courses(state):
    """Coursera에서 특정 주제의 강의를 검색합니다."""
    try:
        # state에서 사용자 질문 추출
        messages = state["messages"]
        query = messages[0].content
        print("=========쿼리========")
        print(query)
        
        encoded_query = quote_plus(query)
        url = f"https://www.coursera.org/search?query={encoded_query}&entityTypeDescription=Courses"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        courses = _extract_courses(response.text, limit=2)
        
        if not courses:
            result = f"'{query}' 관련 Coursera 강의를 찾을 수 없습니다."
        else:
            result = f"'{query}' 관련 Coursera 강의 {len(courses)}개:\n"
            for i, course in enumerate(courses, 1):
                result += f"{i}. {course['name']}\n   - {course['url']}\n"
        
        return {"messages": [AIMessage(content=result)]}
        
    except Exception as e:
        error_message = f"Coursera 검색 중 오류 발생: {str(e)}"
        return {"messages": [AIMessage(content=error_message)]}

   

# # 테스트 코드
# if __name__ == "__main__":
#     from typing import Annotated, Sequence, TypedDict
#     from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
#     from langgraph.graph.message import add_messages

#     class AgentState(TypedDict):
#         messages: Annotated[Sequence[BaseMessage], add_messages]
#    # 테스트용 state 생성
#     test_state = {
#             "messages": [HumanMessage(content="frontend 관련된 강의 추천해줘")]
#         }
    
#     # 함수 실행
#     result = search_coursera_courses(test_state)
    
#    # 결과 출력
#     print("검색 결과:")
#     print(result["messages"][0].content)