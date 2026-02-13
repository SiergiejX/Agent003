import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

app = FastAPI()
llm = ChatOllama(model="llama3", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "agent4_bos_querries")

@app.post("/run")
async def run(payload: dict):
    task = payload.get("input", "")
    response = llm.invoke(f"Generate draft response or summary for this: {task}")
    return {"draft": response.content, "collection": COLLECTION}
