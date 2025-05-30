from fastapi import FastAPI
import uvicorn

app = FastAPI(docs_url="/apis/docs", openapi_url="/apis/openapi.json")

@app.post("/apis/v1/career-path")
def career_path():
    return "커리어추천: cloud 강의 듣기"

@app.post("/apis/v1/rolemodel")
def rolemodel():
    return "rolemodel: cloud 마스터"

@app.post("/apis/v1/career-summary")
def career_summary():
    return "career-summary: 당신은 cloud 최고 전문가 김채연매니저입니다>_<"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)