from agents.main_chatbot.prompt import exception_prompt, intent_prompt, rewrite_prompt, rag_prompt, \
                                        keyword_prompt, chat_summary_prompt, trend_prompt
from agents.main_chatbot.prompt import similar_analysis_prompt, career_recommend_prompt, \
                                       tech_extraction_prompt, future_search_prompt, future_job_prompt,integration_prompt,\
                                       internal_expert_mento_prompt, search_keyword_prompt, external_expert_mento_prompt
from agents.main_chatbot.developstate import DevelopState
from agents.main_chatbot.config import MODEL_NAME, TEMPERATURE, role, skill_set, domain, job
from agents.main_chatbot.response import PromptWrite, PathRecommendResult, RoleModelGroup, GroupedRoleModelResult, SimilarRoadMapResult,TrendResult
from db.postgres import get_company_direction

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI

from vector_store.chroma_search import find_best_match, get_topN_info
from agents.tools.trend_search import trend_analysis_for_keywords, parse_keywords, format_search_results, tavily_search_for_keywords, trend_search
from agents.tools.tavily_search import search_tavily
from agents.tools.lecture_search import lecture_recommend
import json, asyncio

from concurrent.futures import ThreadPoolExecutor
import time

# monitoring
from agents.main_chatbot.performance_monitor import performance_tracker


# pydantic error -> 반복 실행 횟수 설정
def limited_retry_chain(chain, input_data: dict, max_retries: int = 2):
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            print(f"⚠️ Retry {attempt} failed: {e}")
            if attempt == max_retries:
                raise


def clean_brackets(text_or_list):
    """대괄호 제거 헬퍼 함수"""
    if isinstance(text_or_list, list):
        return [item.replace('[', '').replace(']', '') for item in text_or_list]
    elif isinstance(text_or_list, str):
        return text_or_list.replace('[', '').replace(']', '')
    return text_or_list


def exception(state: DevelopState) -> DevelopState:    
    ex_prompt = PromptTemplate(
                    input_variables=["query", "job", "role", "skill_set", "domain"],
                    template=exception_prompt
                    )
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    exception_chain = ex_prompt | llm | StrOutputParser()

    result = exception_chain.invoke({
        "query": state.get('input_query'),
        "job": job,
        "role": role,
        "skill_set": skill_set,
        "domain": domain
    })
    return {
        **state,
        'result': {'text' : result
                   },
        'messages': AIMessage(content=result, name="EXCEPTION")
    }


@performance_tracker("Intent 분석")
def intent_analize(state: DevelopState) -> DevelopState:
    input_query = state['input_query']
    agents=["path_recommend", "role_model", "trend_path", "career_goal", "EXCEPTION"]
    prompt = intent_prompt.format(agents=agents)
    messages = [AIMessage(content=prompt), HumanMessage(input_query)]
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.invoke(messages)
    intent = response.content.strip()
    print(f"Intent: {intent}")
    return {**state, 
            'intent': intent,
            'messages': HumanMessage(input_query)}


@performance_tracker("쿼리 재작성")
def rewrite(state: DevelopState) -> DevelopState:
    try:
        input_query = state.get("input_query", "")
        user_id = state.get("user_id", "")
        career_summary = state['career_summary']
        intent = state['intent']
        chat_summary = state['chat_summary']
        if not input_query:
            return {**state, 
                    'result': {'detail': '입력된 질의가 없습니다.'}}
        if not user_id:
            return {**state, 
                    "result": {'detail': '사용자 ID가 없습니다.'}}
        rewriter_prompt = PromptTemplate(
            input_variables=["career_summary", "chat_summary", "user_query", "role", "skill_set", "domain", "intent"], 
            template=rewrite_prompt
            )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        rewriter_chain = rewriter_prompt | llm.with_structured_output(PromptWrite)
        rewritten_result = rewriter_chain.invoke({
            "career_summary": career_summary,
            "chat_summary" : chat_summary,
            "user_query": input_query,
            "intent": intent,
            # "direction": direction,
            "role": role, "skill_set": skill_set, "domain": domain
        })
        return {
            **state, 
            'rewrited_query': rewritten_result.new_query
        }
    except Exception as e:
        return {**state, 
                'result': {'detail': f"쿼리 재작성 중 오류: {str(e)}"}}


@performance_tracker("RAG 쿼리 재생성")
def ragwrite(state: DevelopState) -> DevelopState:
    """RAG 과정에서 사용할 질의 재생성 노드 (chromaDB version)"""
    query = state.get('rewrited_query', '')
    intent = state.get('intent', '')
    prompt=[HumanMessage(content=rag_prompt.format(query=query, intent=intent))]
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.invoke(prompt)
    return {**state, 
            'rag_query': response.content.strip()  
            }


def similar_roadmap(state:DevelopState) -> DevelopState:
    """
    추출된 사내 구성원 데이터 정보를 바탕으로 공통점 분석하여 로드맵 작성하는 노드
    """
    try: 
        info = find_best_match(state.get('rag_query', ''), state.get('user_id', ''))
        similar_roadmap_prompt = PromptTemplate(
            input_variables=["internal_employee", "career_summary", "user_query", "role", "job", "skill_set"],
            template=similar_analysis_prompt
        )
        parser = PydanticOutputParser(pydantic_object=SimilarRoadMapResult)
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, verbose=True)

        similar_chain = similar_roadmap_prompt | llm | parser
        try:
            result = limited_retry_chain(similar_chain, {
                "internal_employee": info,
                "career_summary": state.get("career_summary", ""),
                "user_query": state.get("rewrited_query", ""),
                "role": role, "skill_set": skill_set, "job": job
            }, max_retries=2)

        except Exception as e:
            print("❌ 모든 재시도 실패. 기본 결과 반환.", e)
            result = SimilarRoadMapResult()

    except Exception as e:
        print("❌ 진행 중 실패. 기본 결과 반환.", e)
        result = SimilarRoadMapResult()
    
    # print(result)
    return {**state,
            'result': result.to_output_dict(),
            'messages': [AIMessage(result.similar_analysis_text),
                         AIMessage(json.dumps(result.to_output_dict()["similar_roadmaps"], ensure_ascii=False))
                        ]
        }

def path(state: DevelopState) -> DevelopState:
    """ 
    추출된 사내 구성원 데이터 정보를 기반으로 경력 증진 경로 추천하는 노드
    현재: 명시적으로 내부에서 노드 순차적 실행하게 만들어 둠
    """
    try:
        # info = find_best_match(state.get('rag_query',''),state.get('user_id',''))
        direction = get_company_direction()
        path_recommend_prompt = PromptTemplate(
        input_variables=["internal_employee", "career_summary", "user_query", "role", "job", "skill_set", "direction"],
        template=career_recommend_prompt
        )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, verbose=True)

        parser = PydanticOutputParser(pydantic_object=PathRecommendResult)
        path_chain = path_recommend_prompt | llm | parser
        try:
            result = limited_retry_chain(path_chain, {
                "career_summary": state.get("career_summary", ""),
                "user_query": state.get("rewrited_query", ""),
                "common_patterns": state.get("result", {}).get("similar_roadmaps", ""),
                "direction": direction,
                "role": role, "skill_set": skill_set, "job": job
            }, max_retries=2)

        except Exception as e:
            print("❌ 모든 재시도 실패. 기본 결과 반환.", e)
            result = PathRecommendResult()

    except Exception as e:
        print("❌ 진행 중 실패. 기본 결과 반환.", e)
        result = PathRecommendResult()

    # print(result)
    return {**state,
            'result': {**state.get("result", {}),
                        **result.to_output_dict()
                      },
            'messages': [AIMessage(result.career_path_text),
                         AIMessage(json.dumps(
                                [r.model_dump() for r in result.career_path_roadmap],
                                ensure_ascii=False))]
            }
    


### role_model 생성 관련 노드 (비동기)
# 비동기 처리
async def run_in_thread(func, *args, **kwargs):
    """동기 함수를 스레드에서 비동기로 실행"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, func, *args, **kwargs)


# 비동기 DB 조회 함수
async def get_topN_info_async(query_text, user_id, top_n, grade=None, years=False):
    """비동기 DB 조회"""
    print(f"🔍 DB 조회 시작: {query_text[:30]}...")
    start_time = time.time()
    
    # 동기 함수를 비동기로 실행
    result = await run_in_thread(get_topN_info, query_text, user_id, top_n, grade, years)
    
    duration = time.time() - start_time
    print(f"✅ DB 조회 완료: {duration:.2f}초")
    return result


async def gpt_generate_with_parser_async(prompt_template, input_vars, parser=None, max_retries=2):
   """GPT API 비동기 호출 + 파싱"""
   for attempt in range(max_retries + 1):
       try:
           # 프롬프트 포맷팅
           formatted_prompt = prompt_template.format(**input_vars)
           
           # GPT API를 스레드에서 비동기로 호출
           def call_gpt():
               llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
               
               if parser:
                   # 파서가 있으면 체인 사용
                   chain = prompt_template | llm | parser
                   result = chain.invoke(input_vars)
                   return result
               else:
                   # 파서가 없으면 단순 텍스트 호출
                   response = llm.invoke([{"role": "user", "content": formatted_prompt}])
                   return response.content.strip()
           
           api_start = time.time()
           result = await run_in_thread(call_gpt)
           api_duration = time.time() - api_start
           print(f"  📡 GPT API 호출: {api_duration:.2f}초")
           
           return result
               
       except Exception as e:
           if attempt < max_retries:
               print(f"⚠️ Retry {attempt + 1} failed: {e}")
               await asyncio.sleep(1)
           else:
               raise e

     
@performance_tracker("사내 전문가 멘토 생성")
async def create_internal_expert(state: DevelopState) -> DevelopState:
   """비동기 사내 전문가 멘토 생성"""
   try:
       print('🔥 사내 전문가 멘토 생성 시작 (비동기)')
       start_time = time.time()
       
       user_query = state.get('rag_query', state.get('input_query', ''))
       print(user_query)
       
       # 1. 비동기 DB 조회
       db_start = time.time()
       expert_info = await get_topN_info_async(
           query_text=user_query,
           user_id=state.get('user_id', ''),
           grade='CL4',
           top_n=5
       )
       db_duration = time.time() - db_start
       print(f"  🗄️ DB 조회: {db_duration:.2f}초")
       
       if not expert_info.strip():
           return None
               
       # 2. 프롬프트 준비
       similar_prompt = PromptTemplate(
           input_variables=["user_query", "internal_employees", "total_count", "skill_set", "role", "job", "domain"],
           template=internal_expert_mento_prompt
       )
       parser = PydanticOutputParser(pydantic_object=RoleModelGroup)
       
       # 3. 비동기 GPT 호출
       gpt_start = time.time()
       result = await gpt_generate_with_parser_async(
           similar_prompt,
           {
               "user_query": user_query,
               "internal_employees": expert_info,
               "total_count": 5,
               "skill_set": skill_set,
               "role": role,
               "domain": domain,
               "job": job
           },
           parser,
           max_retries=2
       )
       gpt_duration = time.time() - gpt_start
       print(f"  🤖 GPT 처리: {gpt_duration:.2f}초")
       
    #    print(result)
       result.group_name = "사내 전문가"
       result.common_project = clean_brackets(result.common_project)

       total_duration = time.time() - start_time
       print(f"✅ 사내 전문가 멘토 생성 완료: {total_duration:.2f}초")
       return result
   
   except Exception as e:
       print(f"❌ 사내 전문가 멘토 생성 실패: {e}")
       return None


@performance_tracker("사내 유사 경력 멘토 생성")
async def create_internal_similar(state: DevelopState) -> DevelopState:
   """비동기 사내 유사 경력 멘토 생성"""
   try:
       print('🔥 사내 유사 경력 멘토 생성 시작 (비동기)')
       start_time = time.time()
       
       user_query = state.get('rag_query', state.get('input_query', ''))
       
       # 1. 비동기 DB 조회
       similar_info = await get_topN_info_async(
           query_text=user_query,
           user_id=state.get('user_id', ''),
           years=True,
           top_n=5
       )
       
       if not similar_info.strip():
           return None
               
       # 2. 프롬프트 준비
       similar_prompt = PromptTemplate(
           input_variables=["user_query", "internal_employees", "total_count", "skill_set", "role", "job", "domain"],
           template=internal_expert_mento_prompt
       )
       parser = PydanticOutputParser(pydantic_object=RoleModelGroup)
       
       # 3. 비동기 GPT 호출
       result = await gpt_generate_with_parser_async(
           similar_prompt,
           {
               "user_query": user_query,
               "internal_employees": similar_info,
               "total_count": 5,
               "skill_set": skill_set,
               "role": role,
               "domain": domain,
               "job": job
           },
           parser,
           max_retries=2
       )
       
       result.group_name = "사내 유사 경력 구성원"
       result.common_project = clean_brackets(result.common_project)

    #    print(result)
       total_duration = time.time() - start_time
       print(f"✅ 사내 유사 경력 멘토 생성 완료: {total_duration:.2f}초")
       return result
   
   except Exception as e:
       print(f"❌ 사내 유사 경력 멘토 생성 실패: {e}")
       return None


@performance_tracker("외부 전문가 멘토 생성")
async def create_external_expert(state: DevelopState) -> DevelopState:
   """비동기 외부 전문가 멘토 생성"""
   try:
       print('🔥 외부 전문가 멘토 생성 시작 (비동기)')
       start_time = time.time()
       
       user_query = state.get('rag_query', state.get('input_query', ''))
       
       # 1. 비동기 키워드 생성
       keyword_prompt = PromptTemplate(
           input_variables=["user_query"],
           template=search_keyword_prompt
       )
       
       keyword_start = time.time()
       keywords_text = await gpt_generate_with_parser_async(keyword_prompt, {"user_query": user_query})
       keyword_duration = time.time() - keyword_start
       print(f"  🔍 키워드 생성: {keyword_duration:.2f}초")
       
       if not keywords_text.strip():
           raise ValueError("키워드 생성 실패")

       keywords = keywords_text.strip().split(',')

       # 2. 병렬 검색 실행 (비동기)
       search_start = time.time()
       search_results_list = await tavily_search_for_keywords(keywords)
       search_duration = time.time() - search_start
       print(f"  🌐 웹 검색: {search_duration:.2f}초")

       # 3. 검색 결과 통합
       all_results = []
       for i, results in enumerate(search_results_list):
           if isinstance(results, Exception):
               print(f"❌ 키워드 {i + 1} 검색 실패: {results}")
               continue
           all_results.extend(results)

       if all_results:
           external_info = f"""[외부 전문가 검색 결과]
검색 키워드: {', '.join(keywords[:5])}
사용자 질문: {user_query}

=== 검색 결과 ===
{chr(10).join(all_results)}
"""
       else:
           external_info = f"""[외부 전문가 기본 정보]
사용자 질문: {user_query}
검색 결과가 제한적이므로 기본 정보로 멘토 생성 필요
"""

       # 4. 비동기 외부 전문가 멘토 생성
       external_prompt = PromptTemplate(
           input_variables=["user_query", "skill_set", "domain", "job", "external_info"],
           template=external_expert_mento_prompt
       )
       parser = PydanticOutputParser(pydantic_object=RoleModelGroup)

       gpt_start = time.time()
       result = await gpt_generate_with_parser_async(
           external_prompt,
           {
               "user_query": user_query,
               "skill_set": skill_set,
               "domain": domain,
               "job": job,
               "external_info": external_info
           },
           parser,
           max_retries=2
       )
       gpt_duration = time.time() - gpt_start
       print(f"  🤖 GPT 처리: {gpt_duration:.2f}초")
       
       result.group_name = "외부 전문가"
       result.real_info = []
       
       total_duration = time.time() - start_time
       print(f"✅ 외부 전문가 멘토 생성 완료: {total_duration:.2f}초")
       return result

   except Exception as e:
       print(f"❌ 외부 전문가 멘토 생성 실패: {e}")
       return None

@performance_tracker("롤모델 생성")
async def role_model(state: DevelopState) -> DevelopState:
   """
   롤모델 그룹 생성 (비동기 병렬 처리)
   Ver.4 완전한 비동기 병렬 생성으로 성능 최적화
   """
   try:
       # 모든 태스크를 병렬로 실행
       print("🚀 멘토 생성 태스크들을 병렬로 시작... (GPT 비동기)")
       parallel_start = time.time()
       
       results = await asyncio.gather(
           create_internal_expert(state),
           create_internal_similar(state), 
           create_external_expert(state),
           return_exceptions=True
       )
       
       parallel_duration = time.time() - parallel_start
       print(f"🔄 병렬 실행 완료: {parallel_duration:.2f}초")
       
       # 성공한 결과만 수집
       successful_groups = []
       for i, result in enumerate(results):
           if isinstance(result, Exception):
               print(f"❌ 태스크 {i+1} 실행 중 예외 발생: {result}")
               continue
           if result is not None:
               successful_groups.append(result)
       
       # 최종 결과 구성
       if successful_groups:
           final_result = GroupedRoleModelResult(
               analysis_summary=f"총 {len(successful_groups)}개의 멘토 그룹이 생성되었습니다.",
               groups=successful_groups
           )
       else:
           print("❌ 생성된 멘토 그룹이 없습니다.")
           final_result = GroupedRoleModelResult()
       
       # 기존 포맷으로 변환
       role_model_list = []
       for i, group in enumerate(final_result.groups):
           role_model_dict = {
               'group_id': f'group_{i+1}',
               'group_name': group.group_name,
               'current_position': group.role_model.current_position,
               'experience_years': group.role_model.experience_years,
               'main_domains': group.role_model.main_domains,
               'advice_message': group.role_model.advice_message,
               'common_skill_set': group.common_skill_set,
               'common_career_path': group.common_career_path,
               'common_project': group.common_project,
               'common_experience': group.common_experience,
               'common_cert': group.common_cert
           }
           role_model_list.append(role_model_dict)
       
       # 요약 텍스트 생성
       if role_model_list:
           summary_text = f"""🎯 진짜 병렬 멘토 생성 완료 (GPT 비동기)

📋 생성된 롤모델 그룹 ({len(role_model_list)}개):
{chr(10).join([f"• {group.group_name}: {group.role_model.name}" for group in final_result.groups])}

💡 {final_result.analysis_summary}
⚡ 병렬 실행 시간: {parallel_duration:.2f}초

📊 상세 롤모델 정보:
{chr(10).join([
   f"▶ {model['group_name']} (ID: {model['group_id']})" + chr(10) +
   f"  • 현재 직책: {model['current_position']}" + chr(10) +
   f"  • 경력: {model['experience_years']}" + chr(10) +
   f"  • 주요 도메인: {', '.join(model['main_domains'])}" + chr(10) +
   f"  • 조언: {model['advice_message']}" + chr(10) +
   f"  • 공통 기술 스택: {', '.join(model['common_skill_set'])}" + chr(10) +
   f"  • 공통 경력 증진 패턴: {', '.join(model['common_career_path'])}" + chr(10) +
   f"  • 공통 프로젝트: {', '.join(model['common_project'])}" + chr(10) +
   f"  • 공통 경험: {', '.join(model['common_experience'])}" + chr(10) +
   f"  • 공통 자격증: {', '.join(model['common_cert'])}" + chr(10)
   for model in role_model_list
])}
"""
       else:
           summary_text = "❌ 멘토 그룹 생성에 실패했습니다."
           
   except Exception as e:
       print(f"❌ 롤모델 생성 중 전체 오류 발생: {e}")
       return {**state, 'error': f'롤모델 생성 중 오류 발생: {str(e)}'}

   return {
       **state,
       'result': {
           'rolemodels': role_model_list,
       },
       'messages': AIMessage(summary_text)
   }


async def trend(state: DevelopState) -> DevelopState:
    """
    기술 트렌드 검색과 사내 교육 추천을 통합하는 최종 함수
    """
    try:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        user_query = state.get('input_query')
        career_summary = state.get("career_summary")

        
        # 1. 병렬로 기술 트렌드 검색과 사내 교육 추천 실행
        trend_analysis, lecture_recommendation = await asyncio.gather(
            trend_search(user_query),             
            lecture_recommend(user_query,career_summary)     
        )

        trend_result = trend_analysis.get('trend_result', '')
        internal_course = lecture_recommendation.get('internal_course', '')
        ax_college = lecture_recommendation.get('ax_college', '')
        explanation = lecture_recommendation.get('explanation', '')
        
        # 4. 통합 분석 실행
        integration_template = PromptTemplate(
            input_variables=["career_summary", "trend_result", "internal_course","ax_college", "explanation"],
            template=integration_prompt
        )
        
        integration_chain = integration_template | llm
        final_result = integration_chain.invoke({
            "career_summary": state.get("career_summary"),
            "trend_result": trend_result,
            "internal_course": internal_course,
            "ax_college": ax_college,
            "explanation": explanation
        })
        
        return {
            **state,
            'result': {'text': final_result.content,
                       'ax_college': ax_college},
            'messages': AIMessage(final_result.content)
        }
        
        
    except Exception as e:
        # 오류 처리
        error_message = f"통합 분석 중 오류 발생: {str(e)}"
        return {
            **state,
            'result': {'text': error_message,
                       'ax_college': error_message},
            'messages': AIMessage(error_message)
        }

    

async def future_career_recommend(state: DevelopState) -> DevelopState:
    """
    경력 요약 기반 미래 직무 추천 노드 (재시도 로직 포함)
    1. 경력 요약, 회사 방향성에서 주요 기술 스택 파악
    2. 다중 소스 검색으로 최신 기술 트렌드 수집 (최대 2회 재시도)
    3. 15년 후 직무 추천
    """
    
    async def search_with_retry(keywords: list, max_retries: int = 2) -> tuple[str, bool]:
        """검색 및 포맷팅을 재시도하는 함수"""
        last_error = None
        
        for attempt in range(max_retries + 1):  # 최초 1회 + 재시도 2회 = 총 3회
            try:
                print(f"검색 시도 {attempt + 1}회차...")
                
                # 검색 실행
                search_results = await trend_analysis_for_keywords(keywords)
                
                # 검색 결과가 비어있는지 확인
                if not search_results or len(search_results) == 0:
                    raise ValueError("검색 결과가 비어있습니다")
                
                # 포맷팅 시도
                formatted_results = format_search_results(search_results)
                
                # 포맷팅 결과가 의미있는지 확인
                if not formatted_results or len(formatted_results.strip()) < 50:
                    raise ValueError("포맷팅된 결과가 너무 짧습니다")
                
                print(f"검색 성공! (시도 {attempt + 1}회차)")
                return formatted_results, True  # 성공 시 (결과, True) 반환
                
            except Exception as e:
                last_error = e
                print(f"검색 시도 {attempt + 1}회차 실패: {str(e)}")
                
                if attempt < max_retries:
                    print(f"{2-attempt}회 더 재시도합니다...")
                    await asyncio.sleep(1)  # 1초 대기 후 재시도
                else:
                    print("모든 재시도 실패")
        
        # 모든 재시도 실패 시
        return f"검색 중 오류 발생 (총 {max_retries + 1}회 시도): {str(last_error)}", False
    
    try:
        # 회사 방향성 주입
        # direction = '모든 업무와 프로젝트에 AI를 기본 적용할 줄 아는 AI 기본 역량을 갖춘 사내 구성원'
        direction = get_company_direction()

        career_summary = state.get('career_summary', '')
        user_query = state.get('input_query', '')
        
        if not career_summary or career_summary == "None":
            return {
                **state,
                'result': {
                    'text': '경력 정보가 없어서 미래 직무를 추천할 수 없습니다. 경력 정보를 먼저 등록해주세요.'
                },
                'messages': AIMessage('경력 정보가 없어서 미래 직무를 추천할 수 없습니다.')
            }
        
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        
        # 1단계: 경력 요약에서 기술 스택 추출
        tech_extraction_template = PromptTemplate(
            input_variables=["career_summary", "user_query", "direction"],
            template=tech_extraction_prompt
        )
        tech_extraction_chain = tech_extraction_template | llm
        tech_analysis_result = tech_extraction_chain.invoke({
            "career_summary": career_summary,
            "user_query": user_query,
            "direction": direction
        })

        # 2단계: 키워드 추출
        keyword_template = PromptTemplate(
            input_variables=["analysis_result"],
            template=future_search_prompt
        )
        keyword_chain = keyword_template | llm
        keywords_result = keyword_chain.invoke({
            "analysis_result": tech_analysis_result.content
        })

        # 3단계: 키워드 기반 최신 내용 검색 (재시도 로직 포함)
        extracted_keywords = [kw.strip() for kw in keywords_result.content.split(',')]
        
        # 재시도 로직으로 검색 실행
        formatted_search_results, search_success = await search_with_retry(extracted_keywords, max_retries=2)
        
        # 검색 결과가 없으면 특별 메시지 반환
        if not search_success:
            return {
                **state,
                'result': {
                    'text': '검색된 결과가 없습니다. 더 많은 정보를 포함하여 질의를 보내주세요.'
                },
                'messages': AIMessage('검색된 결과가 없습니다. 더 많은 정보를 포함하여 질의를 보내주세요.')
            }

        # 4단계: 15년 후 미래 직무 추천
        future_job_template = PromptTemplate(
            input_variables=["career_summary", "tech_analysis", "search_tech_trends", "direction"],
            template=future_job_prompt
        )
        future_job_chain = future_job_template | llm
        final_result = future_job_chain.invoke({
            "career_summary": career_summary,
            "tech_analysis": tech_analysis_result.content,
            "search_tech_trends": formatted_search_results,
            "direction": direction
        })

        return {
            **state,
            'result': {
                'text': final_result.content,
            },
            'messages': AIMessage(final_result.content)
        }
        
    except Exception as e:
        return {
            **state,
            'result': {
                'text': f'미래 직무 추천 중 오류 발생: {str(e)}'
            },
            'error': f'미래 직무 추천 중 오류 발생: {str(e)}'
        }
    
    
def chat_summary(state: DevelopState) -> DevelopState:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    chat_summary_prompttamplate = PromptTemplate(
    input_variables=["chat_summary","user_question","answer"],
    template=chat_summary_prompt
    )
    chat_summary_llm_chain = chat_summary_prompttamplate | llm
    result = chat_summary_llm_chain.invoke({
        "chat_summary": state.get('chat_summary'),
        "user_question": state.get('input_query'),
        "answer": state.get('result')
    })
    return {**state,
        'chat_summary': result.content,
        'messages': AIMessage(result.content)}
