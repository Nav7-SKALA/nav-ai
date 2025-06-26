import os

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

AGENT_ROOT = os.path.join(BASE_DIR, "agents")
AGENT_DIR = {'main_chatbot': os.path.join(AGENT_ROOT, "main_chatbot"),
             'persona_chat': os.path.join(AGENT_ROOT, "persona_chat"),
             'summary': os.path.join(AGENT_ROOT, "summary"),
            }

# 모델 파라미터
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0

### 수행 역할, 기술 스택, 직무 정의

role = ['PM', 'PL', '총괄 PM', '팀장', '사업관리', '사업개발', '기획', '개발자', '개발리더',    
     'Front-End Dev.', 'Back-End Dev.', '분석설계개발', '분석/설계', '구축', '테스트', 'MPA 개발', 
    'RMS 개발', 'R2R 개발', '단위 시스템 개발자', 'Application Architect', 'System Architect', 'Technical Architect', 'Cloud Architect', '운영자', 'DBA', '시스템 어드민', 'DB 어드민', 'System Programmer', 'Unix Administrator', 'DB2 DBA', 'System Engineer', 'Cloud Engineer', 'Quality Engineering', '서버/스토리지 기술지원', '스토리지 기술지원', 'CICD', '컨설팅', 'PP모듈 컨설턴트', '신기술적용', '솔루션관리', '제안PM', '수행PM', '사업 개발 및 제안PM', '대내외 사업 Cost 분석 및 계약 담당', '제안PM 및 협상', '딜리버리', '공통PL'] 

skill_set = ['Biz. Consulting', 'Software Dev.', 'Project Mgmt.', 'Manufacturing Eng.', 'Architect', 'AIX', 'Solution Dev.', 'Biz. Supporting', 'Quality Mgmt.', 'Cloud/Infra Eng.']

job = ['기타', 'Middleware/Database Eng.', 'Data Architect', 'Stakeholder Mgmt.', 'Human Resource Mgmt.', 'Back-end Dev.', 'AIX', 'Biz. Solution', 'Front-end Dev.', 'Sales', 'Application PM', 'Technical Architect', '자동화 Eng.', 'Generative AI Dev.', 'New Biz. Dev.', 'ERP_HCM', 'Financial Mgmt.', 'ERP_T&E', 'ERP_SCM', 'Governance & Public Mgmt.', 'Solution PM', 'Generative AI Model Dev.', 'Strategy Planning', 'PMO', 'ERP', 'ESG/SHE', 'Mobile Dev.', 'Application Architect', 'SCM', 'Offshoring Service Professional', 'System/Network Eng.', 'ERP_FCM', 'Domain Expert', 'Data Center Eng.', 'Cyber Security', 'AI/Data Dev.', 'CRM', 'Infra PM', 'Factory 기획/설계', 'Quality Eng.', '지능화 Eng.']

domain = ['유통/물류/서비스', '제2금융', '(제조) 대외', '공공', '미디어/콘텐츠', '통신', '금융',
       '(제조) 대내 Process', '(제조) 대내 Process, 공통', '공통', '공통, 제2금융',
       '공통, 유통/물류/서비스', '공통, 유통/물류/서비스, 통신, 제2금융', '통신, 제2금융', 'Global',
       '(제조) 대내 Hi-Tech', '(제조) 대내 Hi-Tech, 제2금융', '제2금융, 유통/물류/서비스',
       '유통/물류/서비스, 통신', '통신, 유통/물류/서비스', '금융등', '금융,공공', '대외 및 그룹사',
       '미디어/콘텐츠, 유통/물류/서비스', '유통/물류/서비스, 미디어/콘텐츠', '공통, (제조) 대외',
       '(제조) 대외,제1금융', '제1금융, 공공', '공공, 유통/물류/서비스', '(제조) 대외, 유통/물류/서비스',
       '(제조) 대외, 공통, (제조) 대내 Process', '공통, (제조) 대내 Process', '공공, 은행',
       '은행', '의료', '물류', '유통', '보험', '유통, 보험, 공공', '제조', 'SK그룹',
       '유통/.서비스', '유통/서비스', '제1금융', '(제조) 대내 프로세스']