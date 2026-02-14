import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

app = FastAPI()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)

COLLECTION = os.getenv("COLLECTION", "agent4_bos_querries")

@app.post("/run")
async def run(payload: dict):
    task = payload.get("input", "")
    response = llm.invoke(f"Generate draft response or summary for this: {task}")
    return {"draft": response.content, "collection": COLLECTION}
