import os
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from typing import Dict, Any, List, Optional
import json
import time
import uuid
import urllib.request
from datetime import datetime

app = FastAPI()

# OpenAI-compatible request models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    timeout=120.0,
    temperature=0.3,
    num_ctx=2048
)

# Qdrant connection
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Available collections
COLLECTIONS = {
    "conversations": "agent1_conversations",
    "turns": "agent1_turns", 
    "knowledge": "BazaWiedzy",
    "analytics_queries": "agent3_analitics"
}


def save_query_to_collection(query: str, answer: str, elapsed_time: float, stats_context: Dict[str, Any]):
    """Save query and result to agent3_analitics collection."""
    try:
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Generate embedding for query
        query_vector = generate_embedding(query)
        
        # Prepare payload
        payload = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "response_time_sec": round(elapsed_time, 2),
            "stats_snapshot": {
                "total_conversations": stats_context.get("total_conversations", 0),
                "total_turns": stats_context.get("total_turns", 0),
                "total_documents": stats_context.get("total_documents", 0)
            },
            "model": CHAT_MODEL,
            "agent": "agent3_analytics"
        }
        
        # Save to Qdrant
        point = PointStruct(
            id=hash(query_id) % (10 ** 8),  # Convert UUID to int
            vector=query_vector,
            payload=payload
        )
        
        client.upsert(
            collection_name=COLLECTIONS["analytics_queries"],
            points=[point]
        )
        
        return True
    except Exception as e:
        print(f"Error saving query to collection: {e}")
        return False


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using native Ollama embeddings API."""
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    request = urllib.request.Request(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("embedding", [])


def analyze_conversations(query: str = "") -> Dict[str, Any]:
    """Analyze conversations based on query."""
    try:
        points = client.scroll(
            collection_name=COLLECTIONS["conversations"],
            limit=1000,
            with_payload=True
        )[0]
        
        if not points:
            return {"error": "No conversations found"}
        
        conversations = [p.payload for p in points]
        
        # Basic statistics
        total = len(conversations)
        by_category = {}
        by_channel = {}
        by_resolved = {"resolved": 0, "unresolved": 0}
        by_resolved_by = {}
        total_duration = 0
        total_turns = 0
        csat_scores = []
        sensitive_count = 0
        
        for conv in conversations:
            cat = conv.get("category", "Unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            
            ch = conv.get("channel", "Unknown")
            by_channel[ch] = by_channel.get(ch, 0) + 1
            
            if conv.get("resolved"):
                by_resolved["resolved"] += 1
                resolver = conv.get("resolved_by", "unknown")
                by_resolved_by[resolver] = by_resolved_by.get(resolver, 0) + 1
            else:
                by_resolved["unresolved"] += 1
            
            total_duration += conv.get("duration_sec", 0)
            total_turns += conv.get("turn_count", 0)
            
            if conv.get("csat"):
                csat_scores.append(conv["csat"])
            
            if conv.get("contains_sensitive"):
                sensitive_count += 1
        
        avg_duration = total_duration / total if total > 0 else 0
        avg_turns = total_turns / total if total > 0 else 0
        avg_csat = sum(csat_scores) / len(csat_scores) if csat_scores else 0
        
        stats = {
            "total_conversations": total,
            "by_category": by_category,
            "by_channel": by_channel,
            "resolution": by_resolved,
            "resolved_by": by_resolved_by,
            "avg_duration_sec": round(avg_duration, 1),
            "avg_duration_min": round(avg_duration / 60, 1),
            "avg_turns": round(avg_turns, 1),
            "avg_csat": round(avg_csat, 2),
            "csat_count": len(csat_scores),
            "sensitive_conversations": sensitive_count
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


def analyze_turns(query: str = "") -> Dict[str, Any]:
    """Analyze conversation turns."""
    try:
        points = client.scroll(
            collection_name=COLLECTIONS["turns"],
            limit=1000,
            with_payload=True
        )[0]
        
        if not points:
            return {"error": "No turns found"}
        
        turns = [p.payload for p in points]
        
        total = len(turns)
        by_role = {"student": 0, "bos": 0}
        sentiment_scores = []
        urgency_scores = []
        by_category = {}
        sensitive_count = 0
        
        for turn in turns:
            role = turn.get("role", "unknown")
            if role in by_role:
                by_role[role] += 1
            
            if "sentiment" in turn:
                sentiment_scores.append(turn["sentiment"])
            if "urgency" in turn:
                urgency_scores.append(turn["urgency"])
            
            cat = turn.get("category", "Unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            
            if turn.get("contains_sensitive"):
                sensitive_count += 1
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        avg_urgency = sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0
        
        stats = {
            "total_turns": total,
            "by_role": by_role,
            "by_category": by_category,
            "avg_sentiment": round(avg_sentiment, 3),
            "avg_urgency": round(avg_urgency, 3),
            "sentiment_range": {
                "min": round(min(sentiment_scores), 3) if sentiment_scores else 0,
                "max": round(max(sentiment_scores), 3) if sentiment_scores else 0
            },
            "urgency_range": {
                "min": round(min(urgency_scores), 3) if urgency_scores else 0,
                "max": round(max(urgency_scores), 3) if urgency_scores else 0
            },
            "sensitive_turns": sensitive_count
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


def analyze_knowledge_base(query: str = "") -> Dict[str, Any]:
    """Analyze knowledge base."""
    try:
        points = client.scroll(
            collection_name=COLLECTIONS["knowledge"],
            limit=1000,
            with_payload=True
        )[0]
        
        if not points:
            return {"error": "No documents found"}
        
        documents = [p.payload for p in points]
        
        total = len(documents)
        by_category = {}
        by_file_type = {}
        total_size = 0
        
        for doc in documents:
            cat = doc.get("category", "Unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            
            ft = doc.get("file_type", "txt")
            by_file_type[ft] = by_file_type.get(ft, 0) + 1
            
            content = doc.get("full_content", "")
            total_size += len(content)
        
        avg_size = total_size / total if total > 0 else 0
        
        stats = {
            "total_documents": total,
            "by_category": by_category,
            "by_file_type": by_file_type,
            "avg_document_size": round(avg_size, 1),
            "total_content_size": total_size
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/tags")
async def list_tags():
    """Ollama-compatible tags endpoint."""
    return {
        "models": [
            {
                "name": "agent3-analytics",
                "model": "agent3-analytics",
                "modified_at": "2026-02-12T00:00:00Z",
                "size": 0,
                "digest": "agent3-analytics",
                "details": {
                    "parent_model": "",
                    "format": "agent",
                    "family": "analytics",
                    "families": ["analytics"],
                    "parameter_size": "0",
                    "quantization_level": "none"
                }
            }
        ]
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models - OpenAI API compatible."""
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
                "parent": None,
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
    """OpenAI-compatible chat completions endpoint."""
    
    # Extract last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    
    query = user_messages[-1].content
    query_lower = query.lower()
    
    start_time = time.time()
    
    try:
        # Get statistics
        conv_stats = analyze_conversations("")
        turn_stats = analyze_turns("")
        kb_stats = analyze_knowledge_base("")
        
        # Prepare context
        context = f"""STATYSTYKI SYSTEMU:

ROZMOWY (całe konwersacje z chatbotem): {conv_stats.get('total_conversations', 0)} total, {conv_stats.get('resolution', {}).get('resolved', 0)} rozwiązanych
Średni czas: {conv_stats.get('avg_duration_min', 0):.1f}min, CSAT: {conv_stats.get('avg_csat', 0):.2f}
Kategorie rozmów: {conv_stats.get('by_category', {})}

WYPOWIEDZI (pojedyncze wiadomości w rozmowach): {turn_stats.get('total_turns', 0)} total
Sentyment: {turn_stats.get('avg_sentiment', 0):.2f}, Pilność: {turn_stats.get('avg_urgency', 0):.2f}

BAZA WIEDZY: {kb_stats.get('total_documents', 0)} dokumentów
Kategorie: {list(kb_stats.get('by_category', {}).keys())}"""

        prompt = f"""Jesteś agentem analitycznym systemu obsługi studentów. Dysponujesz następującymi danymi:

{context}

DEFINICJE - BARDZO WAŻNE:
- ROZMOWA (conversation) = kompletna konwersacja/sesja z chatbotem (od początku do końca)
- WYPOWIEDŹ/TURA (turn) = pojedyncza wiadomość studenta lub bota w ramach rozmowy
- Przykład: 1 rozmowa może zawierać 5-10 wypowiedzi (student pisze, bot odpowiada, itd.)

WAŻNE ZASADY:
1. ODPOWIADAJ na pytania o:
   - Statystyki ROZMÓW (liczba, kategorie, tematy, średnie czasy) - użyj liczby z "ROZMOWY"
   - Statystyki WYPOWIEDZI (liczba wiadomości, sentyment) - użyj liczby z "WYPOWIEDZI"
   - Kategorie zgłoszeń i ich liczebność
   - CSAT, urgency rozmów
   - Dokumenty w bazie wiedzy
   - Analitykę systemu obsługi studentów

2. NIE ODPOWIADAJ na pytania całkowicie spoza zakresu systemu, np.:
   - Matematyka (np. "ile jest 2+2", "oblicz pierwiastek")
   - Historia (np. "kto odkrył Amerykę")
   - Astronomia (np. "kiedy jest rok przestępny")
   - Geografia, nauki ścisłe niezwiązane z systemem
   
   W takim przypadku odpowiedz:
   "Nie mam danych, aby odpowiedzieć na to pytanie. Jestem agentem analitycznym systemu obsługi studentów i dysponuję tylko danymi o rozmowach, kategoriach zgłoszeń i bazie wiedzy uczelni."

3. Odpowiadaj KRÓTKO i konkretnie, podając DOKŁADNE LICZBY z powyższych statystyk

PRZYKŁADY DOBRYCH ODPOWIEDZI:
- "ile rozmów z chatbotem" → użyj liczby z "ROZMOWY" (obecnie: {conv_stats.get('total_conversations', 0)})
- "ile wypowiedzi" → użyj liczby z "WYPOWIEDZI" (obecnie: {turn_stats.get('total_turns', 0)})
- "ile wiadomości" → użyj liczby z "WYPOWIEDZI"
- "ile sesji" → użyj liczby z "ROZMOWY"

Pytanie: {query}

Odpowiedź (zwięzła, z konkretnymi liczbami):"""
        
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            answer = f"{answer}\n\nOdpowiedź wygenerowana przez {CHAT_MODEL}"
        except Exception as e:
            answer = f"OGÓLNE STATYSTYKI\n\n{context}\n\n⚠️ LLM niedostępny: {str(e)}"
        
        elapsed_time = time.time() - start_time
        answer = f"{answer}\n\nCzas odpowiedzi: {elapsed_time:.2f}s"
        
        # Save query and result to collection
        save_query_to_collection(
            query=query,
            answer=answer,
            elapsed_time=elapsed_time,
            stats_context={
                "total_conversations": conv_stats.get('total_conversations', 0),
                "total_turns": turn_stats.get('total_turns', 0),
                "total_documents": kb_stats.get('total_documents', 0)
            }
        )
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "agent3-analytics",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(answer.split()),
                "total_tokens": len(query.split()) + len(answer.split())
            }
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"⚠️ Błąd: {str(e)}\n\nCzas odpowiedzi: {elapsed_time:.2f}s"
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "agent3-analytics",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": error_msg
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


@app.get("/stats")
async def get_stats():
    """Get all available statistics."""
    return {
        "conversations": analyze_conversations(""),
        "turns": analyze_turns(""),
        "knowledge_base": analyze_knowledge_base("")
    }
