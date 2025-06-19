import asyncio
from typing import List, Dict
from tavily_search import search_tavily
from reddit_search import search_reddit
from github_search import search_github

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

# 5. 여러 키워드에 대해 병렬 검색
async def trend_analysis_for_keywords(keywords: List[str]) -> List[Dict]:
    results = await asyncio.gather(*[search_all_sources(k) for k in keywords])
    return results