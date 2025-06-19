from typing import List
import requests

async def search_github(keyword: str) -> List[str]:
    def format_repo_for_llm(repo):
        topics = repo.get('topics', [])
        if isinstance(topics, set):
            topics = list(topics)

        return (
            f"Name: {repo['full_name']}\n"
            f"URL: {repo['html_url']}\n"
            f"Description: {repo['description'] or 'No description'}\n"
            f"Stars: {repo['stargazers_count']}\n"
            f"Language: {repo['language']}\n"
            f"Topics: {', '.join(topics)}\n"
            f"Updated at: {repo['updated_at']}"
        )

    keyword = "machine learning"
    query = f"{keyword} stars:>50 created:>2023-01-01"
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=3"

    headers = {
        "Accept": "application/vnd.github.v3+json"
        # "Authorization": "token YOUR_GITHUB_TOKEN"  # 선택적으로 사용
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    result = "\n\n".join(format_repo_for_llm(repo) for repo in data["items"][:5])
    return result