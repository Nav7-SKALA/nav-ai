from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import feedparser
import urllib.parse

## Seb Search

tavily_tool = TavilySearchResults(max_results=5)

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

## DB Search (hard coding - draft ver.)

@tool
def vectorDB_search(query: str) -> str:
    """vectorDB에서 정보 검색 도구"""
    return {"similar_id": ["1", "2", "3"]}

@tool
def RDB_search(member_id: str) -> str:
    """RDB에서 정보 검색 도구"""
    return {"member_id": member_id,
            "information": """
#### 프로젝트 정보 #### 
프로젝트 명: AI 기반 고객 이탈 예측 시스템
프로젝트 설명: 고객 행동 데이터를 분석하여 이탈 가능성을 예측하는 머신러닝 모델 개발
프로젝트 기간: 2023-01~2023-06

프로젝트 명: Generative AI 문서 요약 서비스
프로젝트 설명: GPT API를 활용해 계약서 및 업무 문서를 자동으로 요약하는 웹 서비스 구축
프로젝트 기간: 2023-07~2023-12

프로젝트 명: LLM 기반 사내 Q&A 봇
프로젝트 설명: LangChain과 VectorDB를 활용하여 사내 문서 기반 질문 응답 시스템 개발
프로젝트 기간: 2024-02~2024-04


#### 경험 정보 ####
경험 이름: AI 모델 성능 개선
경험 설명: 기존 의사결정트리 모델을 LSTM 시계열 모델로 개선하여 예측 정확도 향상
경험한 날짜: 2023-05

경험 이름: 벡터 검색 최적화
경험 설명: FAISS 설정 튜닝으로 대용량 벡터 검색 속도 40% 개선
경험한 날짜: 2024-01


#### 기술 스택 정보 #### 
LangChain
PyTorch
OpenAI API
ChromaDB
FastAPI
Docker


#### 자격증 정보 ####
자격증 이름: TensorFlow Developer Certificate
취득일: 2023-08

자격증 이름: ADsP
취득일: 2024-03            
"""}

