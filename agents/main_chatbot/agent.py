from agents.main_chatbot.prompt import exception_prompt, intent_prompt, rewrite_prompt, rag_prompt, \
                                        role_prompt, keyword_prompt, chat_summary_prompt
from agents.main_chatbot.prompt import similar_analysis_prompt, career_recommend_prompt, \
                                       tech_extraction_prompt, future_search_prompt, future_job_prompt, integration_prompt
from agents.main_chatbot.developstate import DevelopState
from agents.main_chatbot.config import MODEL_NAME, TEMPERATURE, role, skill_set, domain, job
from agents.main_chatbot.response import PromptWrite, PathRecommendResult, GroupedRoleModelResult, SimilarRoadMapResult, \
                                        TrendResult
from db.postgres import get_company_direction

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from vector_store.chroma_search import find_best_match, get_topN_info, get_topN_emp
from agents.tools.trend_search import trend_analysis_for_keywords, parse_keywords, format_search_results,trend_search
from agents.tools.lecture_search import lecture_recommend
import json, asyncio


def exception(state: DevelopState) -> DevelopState:    
    ex_prompt = PromptTemplate(
                    input_variables=["query"],
                    template=exception_prompt
                    )
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    exception_chain = ex_prompt | llm | StrOutputParser()

    result = exception_chain.invoke({
        "query": state.get('input_query')
    })
    return {
        **state,
        'result': {'text' : result
                   },
        'messages': AIMessage(content=result, name="EXCEPTION")
    }

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
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        similar_chain = similar_roadmap_prompt | llm.with_structured_output(SimilarRoadMapResult)
        result = similar_chain.invoke({
            "internal_employee": info,
            "career_summary": state.get("career_summary", ""),
            "user_query": state.get("rewrited_query", ""),
            "role": role, "skill_set": skill_set, "job": job
        })
        return {
            **state,
            'result': {'similar_text': result.similar_analysis_text,
                       'similar_roadmaps': result.similar_analysis_roadmap},
            'messages': [AIMessage(result.similar_analysis_text),
                         AIMessage(json.dumps(result.similar_analysis_roadmap, ensure_ascii=False))]
        }
    except Exception as e:
        return {**state,
                'result': {'text': "invoke"+str(e),
                           'similar_roadmap': None},
                'error': f"유사한 사내 구성원 데이터 기반 로드맵 작성 중 오류: {str(e)}"
                }

def path(state: DevelopState) -> DevelopState:
    """ 
    추출된 사내 구성원 데이터 정보를 기반으로 경력 증진 경로 추천하는 노드
    현재: 명시적으로 내부에서 노드 순차적 실행하게 만들어 둠
    """
    try:
        # info = find_best_match(state.get('rag_query',''),state.get('user_id',''))
        direction = '모든 업무와 프로젝트에 AI를 기본 적용할 줄 아는 AI 기본 역량을 갖춘 사내 구성원' # get_company_direction()
        path_recommend_prompt = PromptTemplate(
        input_variables=["internal_employee", "career_summary", "user_query", "role", "job", "skill_set", "direction"],
        template=career_recommend_prompt
        )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        path_chain = path_recommend_prompt | llm.with_structured_output(PathRecommendResult)
        result = path_chain.invoke({
            "career_summary": state.get("career_summary", ""),
            "user_query": state.get("rewrited_query", ""),
            "common_patterns": state.get("result", {}).get("similar_roadmaps", ""),
            "direction": direction,
            "role": role, "skill_set": skill_set, "job": job
        })
        return {
            **state,
            'result': {**state.get("result", {}),
                        'text': result.career_path_text,
                        'roadmaps': result.career_path_roadmap
                        },
            'messages': [AIMessage(result.career_path_text),
                         AIMessage(json.dumps(result.career_path_roadmap, ensure_ascii=False))]
        }
    except Exception as e:
        return {**state, 
                'result': {'text': "invoke"+ str(e),
                           'roadmap' : None
                           },
                'error' : f"경로 추천 중 오류: {str(e)}"
                           }
    
def role_model(state: DevelopState) -> DevelopState:
    """
    롤모델 그룹 생성 노드
    현재: 명시적으로 내부에서 노드 순차적 실행하게 만들어 둠
    """
    try:
        top_n = 5
        info = get_topN_emp(state.get('rag_query',''), state.get('user_id',''), top_n)
        grouping_prompt = PromptTemplate(
            input_variables=["similar_employees", "user_query", "total_count", "skill_set", "job", "role", "domain"],
            template=role_prompt
            )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        rolemodel_chain = grouping_prompt | llm.with_structured_output(GroupedRoleModelResult)
        structured_result = rolemodel_chain.invoke({
            "similar_employees": info,
            "user_query": state.get('input_query'),
            "total_count": top_n,
            "skill_set": skill_set,
            "role": role,
            "domain": domain,
            "job": job
        })
        
        role_model_list = []
        for i, group in enumerate(structured_result.groups):
            role_model_dict = {
                'group_id': f'group_{i+1}',
                'group_name': group.group_name,
                # 'group_description': group.group_description,
                'current_position': group.role_model.current_position,
                'experience_years': group.role_model.experience_years,
                'main_domains': group.role_model.main_domains,
                # 'skill_set': group.role_model.skill_set,
                'advice_message': group.role_model.advice_message,
                'common_skill_set': group.common_skill_set,
                'common_career_path': group.common_career_path,
                'common_project': group.common_project,
                'common_experience': group.common_experience,
                'common_cert': group.common_cert
            }
            role_model_list.append(role_model_dict)
        
        print(role_model_list)
        # 요약 정보 생성
        summary_text = f"""🎯 분석 완료: {structured_result.total_employees}명의 사원 데이터를 {len(structured_result.groups)}개 그룹으로 분류

📋 생성된 롤모델 그룹:
{chr(10).join([f"• {group.group_name}: {group.role_model.name} ({group.member_count}명)" for group in structured_result.groups])}

💡 {structured_result.analysis_summary}

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
        
        return {
            **state,
            'result': {
                'rolemodels': role_model_list,
            },
            'messages': AIMessage(summary_text)
        }
    except Exception as e:
        return {**state, 'error': f'롤모델 생성 중 오류 발생: {str(e)}'}

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
        print("결과 확인하기")
        print("trend_search: ",trend_analysis)
        print("*"*60)
        print("lecture_recommend: ",lecture_recommendation)
        print("*"*60)

        trend_result = trend_analysis.get('trend_result', '')
        internal_course = lecture_recommendation.get('internal_course', '')
        ax_college = lecture_recommendation.get('ax_college', '')
        explanation = lecture_recommendation.get('explanation', '')
        
        # 4. 통합 분석 실행
        integration_template = PromptTemplate(
            input_variables=["career_summary", "trend_result", "internal_course","ax_college", "explanation"],
            template=integration_prompt
        )
        
        integration_chain = integration_template | llm.with_structured_output(TrendResult)
        final_result = integration_chain.invoke({
            "career_summary": state.get("career_summary"),
            "trend_result": trend_result,
            "internal_course": internal_course,
            "ax_college": ax_college,
            "explanation": explanation
        })
        
        # 5. 최종 결과 반환 (기존 구조 유지)
        return {
            **state,
            'result': {'text': final_result.text,
                        'ax_college': final_result.ax_college},
            'messages': AIMessage(final_result.text)
        }
        
    except Exception as e:
        # 오류 처리
        error_message = f"통합 분석 중 오류 발생: {str(e)}"
        return {
            **state,
            'result': {'text': error_message,
                       'ax_college': final_result.ax_college},
            'messages': AIMessage(error_message)
        }


async def future_career_recommend(state: DevelopState) -> DevelopState:
    """
    경력 요약 기반 미래 직무 추천 노드 (async 버전)
    1. 경력 요약, 회사 방향성에서 주요 기술 스택 파악
    2. 다중 소스 검색으로 최신 기술 트렌드 수집
    3. 15년 후 직무 추천
    """
    try:
        # 회사 방향성 주입
        direction = '모든 업무와 프로젝트에 AI를 기본 적용할 줄 아는 AI 기본 역량을 갖춘 사내 구성원' # get_company_direction()

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

        # 3단계: 키워드 기반 최신 내용 검색
        # 키워드를 리스트로 변환
        extracted_keywords = [kw.strip() for kw in keywords_result.content.split(',')]
        
        # 3단계: 다중 소스 병렬 검색 (GitHub, Reddit, Tavily)
        search_results = await trend_analysis_for_keywords(extracted_keywords)
        
        # 검색 결과를 텍스트로 포맷팅
        formatted_search_results = format_search_results(search_results)

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