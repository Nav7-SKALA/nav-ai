import asyncio
from typing import List, Dict
from agents.tools.tavily_search import search_tavily
from agents.tools.reddit_search import search_reddit
from agents.tools.github_search import search_github
from agents.main_chatbot.prompt import keyword_prompt,trend_prompt

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.main_chatbot.config import MODEL_NAME, TEMPERATURE

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


# 여러 키워드에 대해 병렬 검색 (only tavily)
async def tavily_search_for_keywords(keywords: List[str]) -> List[Dict]:
    results = await asyncio.gather(*[search_tavily(k.strip()) for k in keywords])
    return results

#### For careerGoal agent 
def format_search_results(search_results: List[Dict]) -> str:
    """
    다중 소스 검색 결과를 LLM이 읽기 쉬운 형태로 포맷팅
    """
    formatted_sections = []
    
    for result in search_results:
        keyword = result['keyword']
        github_data = result['github']
        reddit_data = result['reddit']
        tavily_data = result['tavily']
        
        section = f"""
=== {keyword} 관련 기술 트렌드 ===

[GitHub 트렌드]
{format_github_data(github_data)}

[Reddit 커뮤니티 논의]
{format_reddit_data(reddit_data)}

[웹 뉴스 및 기사]
{format_tavily_data(tavily_data)}

---
        """
        formatted_sections.append(section)
    
    return "\n".join(formatted_sections)

def format_github_data(github_data: List[str]) -> str:
    """GitHub 검색 결과 포맷팅"""
    if not github_data:
        return "관련 GitHub 프로젝트 정보 없음"
    
    formatted = []
    for i, item in enumerate(github_data[:5]):  # 상위 5개만
        formatted.append(f"• {item}")
    
    return "\n".join(formatted)

def format_reddit_data(reddit_data: List[str]) -> str:
    """Reddit 검색 결과 포맷팅"""
    if not reddit_data:
        return "관련 Reddit 논의 없음"
    
    formatted = []
    for i, item in enumerate(reddit_data[:5]):  # 상위 5개만
        formatted.append(f"• {item}")
    
    return "\n".join(formatted)

def format_tavily_data(tavily_data: List[str]) -> str:
    """Tavily 검색 결과 포맷팅"""
    if not tavily_data:
        return "관련 웹 기사 없음"
    
    formatted = []
    for i, item in enumerate(tavily_data[:5]):  # 상위 5개만
        formatted.append(f"• {item}")
    
    return "\n".join(formatted)

async def trend_search(user_query:str) -> Dict:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

    # 키워드 추출
    keyword_prompttamplate = PromptTemplate(input_variables=["messages"],
                                            template=keyword_prompt
                                            )
    keyword_llm_chain = keyword_prompttamplate | llm
    keywords = parse_keywords(
                (keyword_llm_chain.invoke({"messages": user_query})).content
                )
    # print("***키워드 확인해보자: ", keywords)
    trend_keyword = await trend_analysis_for_keywords(keywords)

    # 기술 검색 결과 분석
    trend_prompttamplate = PromptTemplate(input_variables=["messages", "keyword_result"],
                                          template=trend_prompt
                                          )
    trend_llm_chain = trend_prompttamplate | llm
    result = trend_llm_chain.invoke({
                    "messages": user_query,
                    "keyword_result": trend_keyword,
                })
    
    return {
            'trend_result': result.content
            }