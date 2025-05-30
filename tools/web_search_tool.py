from langchain.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

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

# 사용 예시 (Agent에서 호출될 때)
if __name__ == "__main__":
    # .env 파일에 TAVILY_API_KEY=YOUR_TAVILY_API_KEY 추가 필요
    result = web_search("2024년 최신 AI 컨퍼런스")
    print(result)