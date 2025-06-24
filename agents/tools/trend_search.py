import asyncio
from typing import List, Dict
from agents.tools.tavily_search import search_tavily
from agents.tools.reddit_search import search_reddit
from agents.tools.github_search import search_github

async def search_all_sources(keyword: str) -> Dict[str, List[str]]:
    github, reddit, tavily = await asyncio.gather(
        search_github(keyword),
        search_reddit(keyword),
        search_tavily(keyword)
    )
    return {
        "keyword": keyword,
        "github": github,
        "reddit": reddit,
        "tavily": tavily
    }

# 여러 키워드에 대해 병렬 검색
async def trend_analysis_for_keywords(keywords: List[str]) -> List[Dict]:
    results = await asyncio.gather(*[search_all_sources(k) for k in keywords])
    return results

def parse_keywords(raw_output: str) -> list:
    """
    LLM 출력 결과에서 키워드 목록을 파싱하는 함수.
    각 줄을 키워드로 인식하며 공백이나 빈 줄은 제외.
    """
    return [line.strip() for line in raw_output.strip().splitlines() if line.strip()]