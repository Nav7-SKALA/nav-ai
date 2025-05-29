from fastapi import FastAPI

app = FastAPI(docs_url='/apis/docs', Swagger_url="/apis/openai.json")


@app.get("/")
async def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)