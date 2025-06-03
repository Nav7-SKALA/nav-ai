import re
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus
import requests
from langchain.tools import tool

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