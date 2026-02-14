import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

app = FastAPI()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)

COLLECTION = os.getenv("COLLECTION", "agent5_audit")

@app.post("/run")
async def run(payload: dict):
    text = payload.get("input", "")
    # Anonimizacja / walidacja
    response = llm.invoke(f"Check data compliance and anonymize: {text}")
    return {"audit": response.content, "collection": COLLECTION}
