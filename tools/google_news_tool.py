import feedparser
from langchain.tools import tool
import urllib.parse

@tool
def google_news_search(query: str) -> str:
    """
    Google News에서 특정 키워드에 대한 최신 기술 트렌드를 검색합니다.
    예시: '인공지능', '블록체인', '사이버 보안'
    """
    # 쿼리 문자열을 URL 인코딩
    encoded_query = urllib.parse.quote(query)
    # Google News RSS 피드 URL (검색 쿼리를 포함)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"

    # feedparser를 사용하여 RSS 피드 파싱
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        return f"'{query}'에 대한 Google News 검색 결과를 찾을 수 없습니다."

    # 최신 뉴스 5개 추출
    news_items = []
    for entry in feed.entries[:5]:
        news_items.append(f"제목: {entry.title}\n링크: {entry.link}\n")

    return "\n".join(news_items)


# if __name__ == "__main__":
#     # 동기 함수 테스트
#     print("\n--- '인공지능' 검색 결과 ---")
#     print(google_news_search("인공지능"))


#     print("\n--- '생성형 AI' 검색 결과 ---")
#     print(google_news_search("생성형 AI"))

