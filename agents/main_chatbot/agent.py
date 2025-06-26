from agents.main_chatbot.prompt import exception_prompt, intent_prompt,rewrite_prompt, rag_prompt, path_prompt,role_prompt, keyword_prompt,chat_summary_prompt, trend_prompt
from agents.main_chatbot.developstate import DevelopState
from agents.main_chatbot.config import MODEL_NAME, TEMPERATURE, role, skill_set, domain, job
from agents.main_chatbot.response import PromptWrite, PathRecommendResult, GroupedRoleModelResult
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from vector_store.chroma_search import find_best_match,get_top5_info
from agents.tools.trend_search import trend_analysis_for_keywords, parse_keywords
import json


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
    agents=["path_recommend", "role_model", "trend_path", "EXCEPTION"]
    prompt = intent_prompt.format(agents=agents)
    messages = [AIMessage(content=prompt), HumanMessage(input_query)]
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.invoke(messages)
    intent = response.content.strip()
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
            input_variables=["career_summary","chat_summary", "user_query", "role", "skill_set", "domain","intent"],
            template=rewrite_prompt
            )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        rewriter_chain = rewriter_prompt | llm.with_structured_output(PromptWrite)
        rewritten_result = rewriter_chain.invoke({
            "career_summary": career_summary,
            "chat_summary" : chat_summary,
            "user_query": input_query,
            "intent": intent,
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
    prompt=[HumanMessage(content=rag_prompt.format(query=query,intent=intent))]
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.invoke(prompt)
    return {**state, 
            'rag_query': response.content.strip()  
            }

def path(state: DevelopState) -> DevelopState:
    """ 
    추출된 사내 구성원 데이터 정보를 기반으로 분석(공통점/경력 변화 등)하는 노드
    현재: 명시적으로 내부에서 노드 순차적 실행하게 만들어 둠
    """
    try:
        info = find_best_match(state.get('rag_query',''),state.get('user_id',''))
        path_recommend_prompt = PromptTemplate(
        input_variables=["internal_employee", "career_summary", "user_query", "role", "job", "skill_set"],
        template=path_prompt
        )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        path_chain = path_recommend_prompt | llm.with_structured_output(PathRecommendResult)
        result = path_chain.invoke({
            "internal_employee": info,
            "career_summary": state.get("career_summary", ""),
            "user_query": state.get("rewrited_query", ""),
            "role": role, "skill_set": skill_set, "job": job
        })
        return {
            **state,
            'result': {'text': result.career_path_text,
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
        info = get_top5_info(state.get('rag_query',''),state.get('user_id',''))
        grouping_prompt = PromptTemplate(
            input_variables=["similar_employees", "user_query", "total_count", "skill_set", "role", "domain"],
            template=role_prompt
            )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
        rolemodel_chain = grouping_prompt | llm.with_structured_output(GroupedRoleModelResult)
        structured_result = rolemodel_chain.invoke({
            "similar_employees": info,
            "user_query": state.get('input_query'),
            "total_count": 5,
            "skill_set": skill_set,
            "role": role,
            "domain": domain
        })
        role_model_list = []
        for i, group in enumerate(structured_result.groups, 1):
            role_model_dict = {
                'group_id': f'group_{i}',
                'group_name': group.group_name,
                # 'group_description': group.group_description,
                # 'member_count': group.member_count,
                # 'role_model_name': group.role_model.name,
                'current_position': group.role_model.current_position,
                'experience_years': group.role_model.experience_years,
                'main_domains': group.role_model.main_domains,
                'tech_stack': group.role_model.tech_stack,
                # 'career_highlights': group.role_model.career_highlights,
                'advice_message': group.role_model.advice_message,
                'common_tech_stack': group.common_tech_stack,
                'common_career_path': group.common_career_path
            }
            role_model_list.append(role_model_dict)
            # 요약 정보 생성
        summary_text = f"""
        🎯 분석 완료: {structured_result.total_employees}명의 사원 데이터를 {len(structured_result.groups)}개 그룹으로 분류

        📋 생성된 롤모델 그룹:
        {chr(10).join([f"• {group.group_name}: {group.role_model.name} ({group.member_count}명)" for group in structured_result.groups])}

        💡 {structured_result.analysis_summary}
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
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    keyword_prompttamplate = PromptTemplate(
    input_variables=["messages"],
    template=keyword_prompt
    )
    keyword_llm_chain = keyword_prompttamplate | llm
    keywords = parse_keywords((keyword_llm_chain.invoke({
        "messages": state.get('input_query')
    })).content)
    trend_keyword = await trend_analysis_for_keywords(keywords)
    trend_prompttamplate = PromptTemplate(
    input_variables=["messages", "keyword_result"],
    template=trend_prompt
    )
    trend_llm_chain = trend_prompttamplate | llm
    result = trend_llm_chain.invoke({
        "messages": state.get('input_query'),
        "keyword_result": trend_keyword,
    })
    return {**state,
        'result': {'text': result.content},
        'messages': AIMessage(result.content)}


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