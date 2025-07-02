import os
from tavily import TavilyClient
from typing import List

async def search_tavily(keyword: str) -> List[str]:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(query=keyword, max_results=3)
    results = []
            
    if response.get("answer"):
        results.append(f"요약: {response['answer']}")

    for result in response.get("results", []):
        formatted_result = f"제목: {result.get('title', 'N/A')}\n"
        formatted_result += f"URL: {result.get('url', 'N/A')}\n"
        formatted_result += f"내용: {result.get('content', 'N/A')}\n"
        if result.get('published_date'):
            formatted_result += f"게시일: {result.get('published_date')}\n"
        formatted_result += "-" * 50
        results.append(formatted_result)
    return results