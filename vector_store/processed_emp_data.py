import pandas as pd
import os

def create_employee_documents():
    # CSV 파일 읽기
    employee_df = pd.read_csv('직원history.csv')
    skillset_df = pd.read_csv('skillset_data.csv')
    
    # 스킬셋 매핑 딕셔너리 생성
    skill_mapping = {}
    job_mapping = {}
    
    for _, row in skillset_df.iterrows():
        code = row['코드']
        skill_name = row['Skill set'].strip('.')
        
        skillset_text = str(row['Skillset-직무연계'])
        if '(' in skillset_text and ')' in skillset_text:
            job_category = skillset_text.split('(')[1].replace(')', '')
        else:
            job_category = skillset_text
        
        skill_mapping[code] = skill_name
        job_mapping[code] = job_category
    
    # 출력 디렉토리 생성
    output_dir = 'emp_docs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 고유번호별로 그룹화
    employee_groups = employee_df.groupby('고유번호')
    
    for employee_id, group in employee_groups:
        project_texts = []
        
        # 프로젝트별로 그룹화 (주요 업무/프로젝트 기준)
        project_groups = group.groupby('주요 업무/프로젝트')
        
        for project_name, project_group in project_groups:
            # 해당 프로젝트의 첫 번째 행 기준으로 정보 가져오기
            first_row = project_group.iloc[0]
            
            # 시작연차와 종료연차를 프로젝트 전체 기간으로 설정
            min_start = project_group['시작연차'].min()
            max_end = project_group['종료연차'].max()
            
            # 스킬 정보 처리 (첫 번째 행 기준)
            skill_codes = []
            skill_columns = ['활용 Skill set 1', '활용 Skill set 2', '활용 Skill set 3', '활용 Skill set 4']
            
            for col in skill_columns:
                if pd.notna(first_row[col]) and str(first_row[col]).strip():
                    skill_codes.append(str(first_row[col]).strip())
            
            # 스킬명과 직무 카테고리 추출
            skill_names = []
            job_categories = []
            
            for code in skill_codes:
                if code in skill_mapping:
                    skill_names.append(skill_mapping[code])
                if code in job_mapping:
                    job_categories.append(job_mapping[code])
            
            skill_names_str = ", ".join(list(set(skill_names))) if skill_names else "정보 없음"
            job_categories_str = ", ".join(list(set(job_categories))) if job_categories else "정보 없음"
            
            # 커리어 임팩트 처리 (프로젝트 전체에서 TRUE 찾기)
            career_impact = ""
            true_rows = project_group[project_group['커리어 형성에 큰 영향을 받은 업무나 시기'] == True]
            if not true_rows.empty:
                impact_desc = true_rows.iloc[0]['큰 영향을 받은 업무/시기에 대한 설명']
                if pd.notna(impact_desc):
                    career_impact = f"특히 커리어 역량에 큰 영향을 받았으며, 이에 대한 설명은 다음과 같습니다, {impact_desc}"
                else:
                    career_impact = "특히 커리어 역량에 큰 영향을 받았습니다."
            elif (project_group['커리어 형성에 큰 영향을 받은 업무나 시기'] == False).any():
                career_impact = "커리어 역량에 큰 영향을 받지 못했습니다."
    
            # content_text 생성 (프로젝트별로 통합)
            content_text = (
                f"**{first_row['고유번호']}** 직원은 {min_start}년차부터 {max_end}년차까지의 경험으로, "
                f"{first_row['Industry/Domain']} 도메인에서 {first_row['수행역할']} 역할을 담당했습니다. "
                f"주요 업무는 {first_row['주요 업무/프로젝트']}이며, "
                f"프로젝트 규모는 {first_row['프로젝트 규모']}입니다. "
                f"활용한 기술 스택은 {skill_names_str}이며, "
                f"관련된 직무 영역은 {job_categories_str}입니다. "
                f"{career_impact + ' ' if career_impact else ''}"
                f"[Grade: C4]"
            )
            
            project_texts.append(content_text)
        
        # 한 사원의 모든 프로젝트를 하나의 파일로 저장
        full_document = "\n\n".join(project_texts)
        filename = f"{output_dir}/{employee_id}_C4.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_document)
    
    print(f"총 {len(employee_groups)}명의 직원 문서가 생성되었습니다.")

if __name__ == "__main__":
    create_employee_documents()