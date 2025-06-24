import praw
import asyncio
import os
from typing import List
def sync_search_reddit(keyword: str) -> List[str]:
        REDDIT = praw.Reddit(
        client_id=os.getenv("REDDIT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent= os.getenv("REDDIT_AGENT")
        )
        results = []
        for submission in REDDIT.subreddit("all").search(keyword, limit=5, sort="relevance", time_filter="week"):
                results.append(f"{submission.title} - {submission.url}")
        return results
async def search_reddit(keyword: str) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_search_reddit, keyword)
