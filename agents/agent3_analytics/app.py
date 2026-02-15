"""
Agent 3 Analytics - Optimized Version
Retrieval Augmented Generation (RAG) based analytics agent
"""
import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Header
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import urllib.request

from rag_engine import RAGEngine

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(title="Agent3 Analytics", version="2.0.0")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

AGENT3_SYSTEM_PROMPT = """Jesteś zaawansowanym agentem analitycznym (Agent 3) odpowiedzialnym za usprawnianie działania chatbotów BOS.

DOSTĘPNE DANE (zanonimizowane, ostatnie 7 dni):
- rozmowy z chatbotem (intent, czas, CSAT, status)
- zdarzenia systemu (retrieval scores, handoffs)
- eskalacje do człowieka z powodami
- feedback użytkowników (CSAT, sentiment)
- błędy techniczne

ZADANIA:
1. Analizuj intenty (jakość, wzrosty, problemy)
2. Identyfikuj przyczyny eskalacji
3. Generuj konkretne rekomendacje ulepszeń

ZASADY:
✅ Używaj TYLKO danych ze źródeł
✅ Podawaj konkretne liczby i metryki
✅ Formatuj odpowiedzi czytelnie (tabele, listy)
✅ Cytuj źródło informacji

❌ NIE wymyślaj danych
❌ NIE zgaduj metryk"""

# ============================================================================
# MODELS
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize LLM
llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    timeout=180.0,
    temperature=0.2,
    num_ctx=2048,  # Reduced from 4096 - GTX 1050 has only 4GB VRAM
    num_gpu=1,  # Use 1 GPU (was -1 which forced CPU mode)
    num_thread=4  # Reduced from 8 to avoid thread overhead
)

# Initialize Qdrant client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Initialize RAG Engine
rag_engine = RAGEngine(
    client=client,
    ollama_base_url=OLLAMA_BASE_URL,
    embedding_model=EMBEDDING_MODEL
)
print("[AGENT3] ✓ Initialized RAG Engine")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama."""
    try:
        payload = {"model": EMBEDDING_MODEL, "prompt": text[:500]}
        request = urllib.request.Request(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("embedding", [0.0] * 768)
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        return [0.0] * 768


def categorize_query(query: str) -> str:
    """Categorize query based on keywords."""
    q = query.lower()
    
    analytics_kw = ["analiza", "raport", "rekomendacje", "wykryj", "intent", 
                   "eskalacj", "usprawn", "wzrost", "jakość"]
    topic_kw = ["temat", "tematy", "najcz", "ranking", "top", "popularn"]
    stats_kw = ["ile", "liczba", "statystyki", "średni", "total", "suma"]
    
    if any(kw in q for kw in analytics_kw):
        return "analytics"
    elif any(kw in q for kw in topic_kw):
        return "topic_discovery"
    elif any(kw in q for kw in stats_kw):
        return "stats_query"
    else:
        return "general"


def save_query_to_collection(
    query: str,
    answer: str,
    elapsed_time: float,
    stats_context: Dict[str, Any],
    query_metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Save query and result to a3_analytics collection."""
    try:
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        query_vector = generate_embedding(query)
        query_category = categorize_query(query)
        metadata = query_metadata or {}
        
        # Quality flags
        has_llm_error = "LLM niedostępny" in answer or "Błąd" in answer
        has_no_data = "Nie mam danych" in answer or "No data" in answer.lower()
        
        # Token estimation
        query_tokens = len(query.split())
        answer_tokens = len(answer.split())
        
        payload = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "response_time_sec": round(elapsed_time, 2),
            "query_category": query_category,
            "query_length": len(query),
            "query_tokens": query_tokens,
            "answer_tokens": answer_tokens,
            "stats_snapshot": {
                "total_conversations": stats_context.get("total_conversations", 0),
                "total_turns": stats_context.get("total_turns", 0),
                "total_documents": stats_context.get("total_documents", 0),
                "data_sources_used": metadata.get("data_sources_used", [])
            },
            "quality_flags": {
                "has_llm_error": has_llm_error,
                "has_no_data": has_no_data,
                "is_complete": not (has_llm_error or has_no_data)
            },
            "analytics_context": {
                "analyzed_collections": metadata.get("collections_queried", []),
                "records_analyzed": metadata.get("total_records", 0),
                "time_range_days": metadata.get("time_range_days", 7)
            },
            "model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "agent": "agent3_analytics",
            "version": "2.0"
        }
        
        point = PointStruct(
            id=hash(query_id) % (10 ** 8),
            vector=query_vector,
            payload=payload
        )
        
        client.upsert(collection_name="a3_analytics", points=[point])
        
        info = client.get_collection("a3_analytics")
        print(f"[AGENT3] ✓ Saved query {query_id[:8]}... to a3_analytics | Total: {info.points_count}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save query: {e}")
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        collections = client.get_collections()
        qdrant_connected = True
        a3_analytics_exists = any(c.name == "a3_analytics" for c in collections.collections)
    except:
        qdrant_connected = False
        a3_analytics_exists = False
    
    return {
        "status": "healthy",
        "agent": "agent3_analytics",
        "qdrant_connected": qdrant_connected,
        "a3_analytics_exists": a3_analytics_exists,
        "model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "version": "2.0.0-rag"
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "agent3-analytics",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agent3",
                "permission": [],
                "root": "agent3-analytics",
                "parent": None
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
    """
    OpenAI-compatible chat completions endpoint with RAG.
    Uses semantic search across Qdrant collections.
    Maintains conversation history for context-aware responses.
    """
    print(f"[AGENT3] ========== CHAT COMPLETIONS CALLED ==========", flush=True)
    
    # Extract conversation history
    conversation_history = []
    for msg in request.messages:
        if msg.role in ["user", "assistant", "system"]:
            conversation_history.append(f"{msg.role.upper()}: {msg.content}")
    
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        print(f"[AGENT3] ERROR: No user message found", flush=True)
        return {"error": "No user message found"}
    
    query = user_messages[-1].content
    start_time = time.time()
    
    print(f"\n{'='*80}", flush=True)
    print(f"[AGENT3] Query: {query}", flush=True)
    print(f"[AGENT3] Conversation history: {len(conversation_history)} messages", flush=True)
    print(f"[AGENT3] DEBUG: Request messages count: {len(request.messages)}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    try:
        print(f"[AGENT3] DEBUG: Calling RAG Engine retrieve_and_generate()...", flush=True)
        print(f"[AGENT3] DEBUG: RAG Engine instance: {rag_engine}", flush=True)
        print(f"[AGENT3] DEBUG: LLM instance: {llm}", flush=True)
        
        # Use RAG pipeline with conversation history
        rag_result = rag_engine.retrieve_and_generate(
            query=query,
            system_prompt=AGENT3_SYSTEM_PROMPT,
            llm_invoke_func=llm.invoke,
            collections=None,  # Auto-select
            time_range_days=7,
            conversation_history=conversation_history  # Include history for context
        )
        
        answer = rag_result["answer"]
        sources = rag_result["sources"]
        metadata = rag_result["metadata"]
        
        # Add metadata to answer
        if sources:
            answer += f"\n\n📚 Źródła: {', '.join(sources)}"
            answer += f"\n🔍 Znaleziono {metadata['total_results']} rekordów"
        
        elapsed_time = time.time() - start_time
        answer += f"\n\n⏱️ Czas: {elapsed_time:.2f}s"
        answer += f"\n🤖 Model: {CHAT_MODEL} (RAG)"
        
        # Save to a3_analytics
        query_metadata = {
            "collections_queried": sources,
            "total_records": metadata.get('total_results', 0),
            "data_sources_used": ["rag_engine"],
            "time_range_days": 7,
            "retrieval_mode": "semantic_search",
            "results_per_collection": metadata.get('results_per_collection', {})
        }
        
        save_query_to_collection(
            query=query,
            answer=answer,
            elapsed_time=elapsed_time,
            stats_context={
                "total_conversations": 0,
                "total_turns": 0,
                "total_documents": metadata.get('total_results', 0)
            },
            query_metadata=query_metadata
        )
        
        print(f"[AGENT3] ✓ Response in {elapsed_time:.2f}s from {len(sources)} sources\n")
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "agent3-analytics-rag",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(answer.split()),
                "total_tokens": len(query.split()) + len(answer.split())
            },
            "rag_metadata": {
                "sources": sources,
                "total_results": metadata.get('total_results', 0),
                "retrieval_success": metadata.get('retrieval_success', False)
            }
        }
        
    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        error_trace = traceback.format_exc()
        print(f"[AGENT3] ❌ ERROR: {error_trace}")
        
        error_msg = f"⚠️ Błąd: {str(e)}\n\n⏱️ Czas: {elapsed_time:.2f}s"
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "agent3-analytics-rag",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": error_msg},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


@app.get("/api/analytics/history")
async def get_analytics_history(limit: int = 50, category: Optional[str] = None):
    """Get query history from a3_analytics."""
    try:
        filter_condition = None
        if category:
            filter_condition = {
                "must": [{"key": "query_category", "match": {"value": category}}]
            }
        
        points = client.scroll(
            collection_name="a3_analytics",
            limit=limit,
            with_payload=True,
            with_vectors=False,
            scroll_filter=filter_condition
        )[0]
        
        queries = []
        for point in points:
            payload = point.payload
            queries.append({
                "query_id": payload.get("query_id"),
                "timestamp": payload.get("timestamp"),
                "query": payload.get("query"),
                "answer": payload.get("answer", "")[:200] + "...",
                "response_time_sec": payload.get("response_time_sec"),
                "category": payload.get("query_category"),
                "quality_flags": payload.get("quality_flags", {}),
                "model": payload.get("model")
            })
        
        return {
            "status": "success",
            "total": len(queries),
            "category_filter": category,
            "queries": queries
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/analytics/stats")
async def get_analytics_stats():
    """Get statistics about query history."""
    try:
        points = client.scroll(
            collection_name="a3_analytics",
            limit=1000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        if not points:
            return {
                "status": "success",
                "total_queries": 0,
                "by_category": {},
                "performance": {},
                "quality": {}
            }
        
        queries = [p.payload for p in points]
        total = len(queries)
        
        # By category
        by_category = {}
        for q in queries:
            cat = q.get("query_category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # Performance
        response_times = [q.get("response_time_sec", 0) for q in queries]
        avg_time = sum(response_times) / len(response_times)
        
        # Quality
        successful = sum(1 for q in queries if q.get("quality_flags", {}).get("is_complete", False))
        
        return {
            "status": "success",
            "total_queries": total,
            "by_category": by_category,
            "performance": {
                "avg_response_time_sec": round(avg_time, 2),
                "min_response_time_sec": round(min(response_times), 2),
                "max_response_time_sec": round(max(response_times), 2)
            },
            "quality": {
                "successful_queries": successful,
                "success_rate_percent": round(successful / total * 100, 1) if total > 0 else 0
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/analytics/search")
async def search_analytics_history(q: str, limit: int = 10):
    """Search query history by text."""
    try:
        query_vector = generate_embedding(q)
        
        import requests
        search_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/a3_analytics/points/search"
        search_payload = {
            "vector": query_vector,
            "limit": limit,
            "score_threshold": 0.5,
            "with_payload": True,
            "with_vector": False
        }
        
        response = requests.post(search_url, json=search_payload, timeout=30)
        response.raise_for_status()
        result_data = response.json()
        search_results = result_data.get("result", [])
        
        queries = []
        for hit in search_results:
            payload = hit.get("payload", {})
            queries.append({
                "score": hit.get("score"),
                "query": payload.get("query"),
                "answer": payload.get("answer", "")[:200] + "...",
                "category": payload.get("query_category"),
                "timestamp": payload.get("timestamp")
            })
        
        return {
            "status": "success",
            "query": q,
            "total_results": len(queries),
            "results": queries
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
