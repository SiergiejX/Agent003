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
    
    "analytics_queries": "agent3_analitics",
    # Agent 3 Analytics collections
    "a3_conversations": "a3_conversations_anon",
    "a3_messages": "a3_messages_anon",
    "a3_events": "a3_events",
    "a3_handoff_cases": "a3_handoff_cases_anon",
    "a3_failures": "a3_failures_anon",
    "a3_feedback": "a3_feedback_anon",
    "a3_intent_catalog": "a3_intent_catalog",
    "a3_sentiment_events": "a3_sentiment_events",
    "a3_ticket_metrics": "a3_ticket_metrics_anon",
    "a3_suggestions": "a3_suggestions"
}

# Agent 3 Analytics System Prompt
AGENT3_SYSTEM_PROMPT = """Jesteś zaawansowanym agentem analitycznym (Agent 3) odpowiedzialnym za usprawnianie działania chatbotów BOS i procesów Biura Obsługi Studenta.

DOSTĘPNE DANE (wyłącznie zanonimizowane):
- a3_conversations_anon: zanonimizowane rozmowy z metadanymi (intent, czas trwania, status rozwiązania)
- a3_messages_anon: zanonimizowane wiadomości (bez treści osobowych)
- a3_events: zdarzenia pipeline (retrieval_score, handoff, błędy)
- a3_handoff_cases_anon: przypadki eskalacji do człowieka
- a3_failures_anon: błędy systemu
- a3_feedback_anon: feedback użytkowników (CSAT, sentiment)

ZAKRES ANALIZY: ostatnie 7 dni

GŁÓWNE ZADANIA ANALITYCZNE:

1. ANALIZA INTENTÓW - wykryj:
   - TOP 5 intentów z największym wzrostem liczby rozmów (porównaj z poprzednim okresem)
   - TOP 5 intentów o najgorszej jakości obsługi:
     * niski retrieval_top_score (<0.5)
     * wysoki handoff_rate (>30%)
     * dużo negatywnego feedbacku (CSAT <3.0)
     * długi czas rozwiązania (>10 min)

2. ANALIZA ESKALACJI:
   - Zidentyfikuj 3 najczęstsze przyczyny eskalacji do człowieka
   - Przeanalizuj powody: brak odpowiedzi w KB, niejasny intent, prośba użytkownika, błąd techniczny

3. REKOMENDACJE:
   - Wygeneruj 5 konkretnych rekomendacji usprawnień:
     * dodanie/aktualizacja dokumentów w bazie wiedzy
     * poprawa rozpoznawania intentów
     * usprawnienia techniczne
     * szkolenia dla operatorów BOS

FORMAT ODPOWIEDZI:

1. PODSUMOWANIE WYKONAWCZE (maksymalnie 10 zdań):
   - Kluczowe metryki systemu
   - Najważniejsze wnioski z analizy
   - Krytyczne problemy wymagające uwagi

2. TABELA METRYK PER INTENT:
   | Intent | Rozmowy | Wzrost% | Avg Score | Handoff% | Avg CSAT | Avg Time |
   |--------|---------|---------|-----------|----------|----------|----------|
   | ...    | ...     | ...     | ...       | ...      | ...      | ...      |

3. TOP 3 PRZYCZYNY ESKALACJI:
   - Przyczyna 1: [opis] (liczba przypadków: X)
   - Przyczyna 2: [opis] (liczba przypadków: Y)
   - Przyczyna 3: [opis] (liczba przypadków: Z)

4. REKOMENDACJE (format JSON):
```json
[
  {
    "recommendation_id": "REC-001",
    "priority": "HIGH|MEDIUM|LOW",
    "category": "KNOWLEDGE_BASE|INTENT_RECOGNITION|TECHNICAL|TRAINING",
    "title": "Krótki tytuł rekomendacji",
    "description": "Szczegółowy opis problemu i sugerowanego rozwiązania",
    "expected_impact": "Oczekiwany wpływ na metryki",
    "effort": "Szacowany nakład pracy: HIGH|MEDIUM|LOW",
    "affected_intents": ["INTENT1", "INTENT2"],
    "implementation_steps": ["Krok 1", "Krok 2", "Krok 3"]
  }
]
```

ZASADY PRACY:

✅ ZAWSZE:
- Bazuj WYŁĄCZNIE na rzeczywistych danych z kolekcji
- Podawaj KONKRETNE LICZBY i metryki
- Używaj zanonimizowanych identyfikatorów (conversation_id_anon, user_id_anon)
- Obliczaj wzrosty procentowe względem poprzedniego okresu
- Weryfikuj wszystkie obliczenia

❌ NIGDY:
- NIE zgaduj brakujących danych
- NIE używaj danych osobowych (imiona, nazwiska, numery studentów)
- NIE dodawaj fikcyjnych metryk
- NIE przekraczaj 10 zdań w podsumowaniu

METRYKI KLUCZOWE:
- retrieval_top_score: podobieństwo do najlepszego dokumentu (0.0-1.0)
- handoff_rate: % rozmów eskalowanych do człowieka
- CSAT: satysfakcja użytkownika (1.0-5.0)
- resolution_time: czas rozwiązania rozmowy (minuty)
- intent_confidence: pewność klasyfikacji intentu (0.0-1.0)

W przypadku braku danych w którejkolwiek kolekcji, wyraźnie to zaznacz i pracuj z dostępnymi danymi."""


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


def perform_deep_analytics() -> Dict[str, Any]:
    """Perform comprehensive analytics on Agent 3 collections for last 7 days."""
    from datetime import datetime, timedelta
    
    results = {
        "data_availability": {},
        "intent_metrics": {},
        "escalation_analysis": {},
        "quality_issues": [],
        "growth_trends": [],
        "error": None
    }
    
    try:
        # Calculate 7-day window
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        
        # Query a3_conversations_anon
        try:
            conversations = client.scroll(
                collection_name=COLLECTIONS["a3_conversations"],
                limit=1000,
                with_payload=True
            )[0]
            results["data_availability"]["conversations"] = len(conversations)
            
            # Analyze by intent
            intent_data = {}
            for conv in conversations:
                payload = conv.payload
                intent = payload.get("primary_intent", "UNKNOWN")
                
                if intent not in intent_data:
                    intent_data[intent] = {
                        "count": 0,
                        "resolved": 0,
                        "total_duration": 0,
                        "durations": [],
                        "handoffs": 0
                    }
                
                intent_data[intent]["count"] += 1
                if payload.get("resolved"):
                    intent_data[intent]["resolved"] += 1
                
                duration = payload.get("duration_min", 0)
                intent_data[intent]["total_duration"] += duration
                intent_data[intent]["durations"].append(duration)
                
                if payload.get("handoff_to_human"):
                    intent_data[intent]["handoffs"] += 1
            
            results["intent_metrics"] = intent_data
            
        except Exception as e:
            results["data_availability"]["conversations"] = f"Error: {str(e)}"
        
        # Query a3_events
        try:
            events = client.scroll(
                collection_name=COLLECTIONS["a3_events"],
                limit=1000,
                with_payload=True
            )[0]
            results["data_availability"]["events"] = len(events)
            
            # Analyze retrieval scores by intent
            retrieval_by_intent = {}
            for event in events:
                payload = event.payload
                intent = payload.get("intent", "UNKNOWN")
                score = payload.get("retrieval_top_score")
                
                if score is not None:
                    if intent not in retrieval_by_intent:
                        retrieval_by_intent[intent] = []
                    retrieval_by_intent[intent].append(score)
            
            results["retrieval_scores"] = retrieval_by_intent
            
        except Exception as e:
            results["data_availability"]["events"] = f"Error: {str(e)}"
        
        # Query a3_handoff_cases_anon
        try:
            handoffs = client.scroll(
                collection_name=COLLECTIONS["a3_handoff_cases"],
                limit=1000,
                with_payload=True
            )[0]
            results["data_availability"]["handoff_cases"] = len(handoffs)
            
            # Analyze handoff reasons
            handoff_reasons = {}
            for handoff in handoffs:
                payload = handoff.payload
                reason = payload.get("handoff_reason", "UNKNOWN")
                handoff_reasons[reason] = handoff_reasons.get(reason, 0) + 1
            
            results["escalation_analysis"]["reasons"] = handoff_reasons
            
        except Exception as e:
            results["data_availability"]["handoff_cases"] = f"Error: {str(e)}"
        
        # Query a3_feedback_anon
        try:
            feedbacks = client.scroll(
                collection_name=COLLECTIONS["a3_feedback"],
                limit=1000,
                with_payload=True
            )[0]
            results["data_availability"]["feedback"] = len(feedbacks)
            
            # Analyze CSAT by intent
            csat_by_intent = {}
            for feedback in feedbacks:
                payload = feedback.payload
                intent = payload.get("intent", "UNKNOWN")
                csat = payload.get("csat_score")
                
                if csat is not None:
                    if intent not in csat_by_intent:
                        csat_by_intent[intent] = []
                    csat_by_intent[intent].append(csat)
            
            results["csat_scores"] = csat_by_intent
            
        except Exception as e:
            results["data_availability"]["feedback"] = f"Error: {str(e)}"
        
        # Query a3_failures_anon
        try:
            failures = client.scroll(
                collection_name=COLLECTIONS["a3_failures"],
                limit=1000,
                with_payload=True
            )[0]
            results["data_availability"]["failures"] = len(failures)
            
        except Exception as e:
            results["data_availability"]["failures"] = f"Error: {str(e)}"
        
        # Calculate combined metrics per intent
        combined_metrics = []
        for intent, data in intent_data.items():
            count = data["count"]
            
            # Calculate averages
            avg_duration = data["total_duration"] / count if count > 0 else 0
            resolution_rate = (data["resolved"] / count * 100) if count > 0 else 0
            handoff_rate = (data["handoffs"] / count * 100) if count > 0 else 0
            
            # Get retrieval scores
            avg_retrieval = 0
            if intent in retrieval_by_intent and retrieval_by_intent[intent]:
                avg_retrieval = sum(retrieval_by_intent[intent]) / len(retrieval_by_intent[intent])
            
            # Get CSAT
            avg_csat = 0
            if intent in csat_by_intent and csat_by_intent[intent]:
                avg_csat = sum(csat_by_intent[intent]) / len(csat_by_intent[intent])
            
            combined_metrics.append({
                "intent": intent,
                "conversations": count,
                "avg_duration_min": round(avg_duration, 2),
                "resolution_rate": round(resolution_rate, 1),
                "handoff_rate": round(handoff_rate, 1),
                "avg_retrieval_score": round(avg_retrieval, 3),
                "avg_csat": round(avg_csat, 2)
            })
        
        # Sort by conversation count (descending)
        combined_metrics.sort(key=lambda x: x["conversations"], reverse=True)
        results["combined_metrics"] = combined_metrics
        
        return results
        
    except Exception as e:
        results["error"] = str(e)
        return results


def save_recommendations_to_collection(recommendations: List[Dict[str, Any]]) -> bool:
    """Save recommendations to a3_suggestions collection."""
    try:
        points = []
        for rec in recommendations:
            rec_id = rec.get("recommendation_id", str(uuid.uuid4()))
            
            # Generate embedding from recommendation text
            text_to_embed = f"{rec.get('title', '')} {rec.get('description', '')}"
            vector = generate_embedding(text_to_embed)
            
            # Prepare payload
            payload = {
                "recommendation_id": rec_id,
                "timestamp": datetime.now().isoformat(),
                "priority": rec.get("priority", "MEDIUM"),
                "category": rec.get("category", "GENERAL"),
                "title": rec.get("title", ""),
                "description": rec.get("description", ""),
                "expected_impact": rec.get("expected_impact", ""),
                "effort": rec.get("effort", "MEDIUM"),
                "affected_intents": rec.get("affected_intents", []),
                "implementation_steps": rec.get("implementation_steps", []),
                "status": "PENDING"
            }
            
            # Create point
            point = PointStruct(
                id=hash(rec_id) % (10 ** 8),
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upsert to collection
        if points:
            client.upsert(
                collection_name=COLLECTIONS["a3_suggestions"],
                points=points
            )
            return True
        return False
        
    except Exception as e:
        print(f"Error saving recommendations: {e}")
        return False


@app.post("/api/analytics/run")
async def run_analytics():
    """Run comprehensive analytics and save recommendations."""
    try:
        analytics_data = perform_deep_analytics()
        
        return {
            "status": "success",
            "data": analytics_data,
            "message": "Analytics completed successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/recommendations/save")
async def save_recommendations(recommendations: List[Dict[str, Any]]):
    """Save recommendations to a3_suggestions collection."""
    try:
        success = save_recommendations_to_collection(recommendations)
        if success:
            return {
                "status": "success",
                "saved": len(recommendations),
                "message": f"Successfully saved {len(recommendations)} recommendations"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to save recommendations"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


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
    
    # Check if this is a deep analytics request
    analytics_keywords = ["analiza", "raport", "rekomendacje", "wykryj", "intent", "eskalacj", "usprawn", 
                         "wzrost", "jakość", "quality", "handoff", "top 5", "przyczyn"]
    is_analytics_request = any(keyword in query_lower for keyword in analytics_keywords)
    
    try:
        if is_analytics_request:
            # Perform deep analytics
            analytics_data = perform_deep_analytics()
            
            # Format analytics data for LLM
            analytics_context = f"""DANE Z OSTATNICH 7 DNI:

DOSTĘPNOŚĆ DANYCH:
- Rozmowy (a3_conversations_anon): {analytics_data['data_availability'].get('conversations', 0)} rekordów
- Zdarzenia (a3_events): {analytics_data['data_availability'].get('events', 0)} rekordów
- Eskalacje (a3_handoff_cases): {analytics_data['data_availability'].get('handoff_cases', 0)} rekordów
- Feedback (a3_feedback): {analytics_data['data_availability'].get('feedback', 0)} rekordów
- Błędy (a3_failures): {analytics_data['data_availability'].get('failures', 0)} rekordów

METRYKI PER INTENT:
"""
            
            # Add combined metrics
            if analytics_data.get('combined_metrics'):
                for metric in analytics_data['combined_metrics'][:10]:  # Top 10
                    analytics_context += f"""
Intent: {metric['intent']}
  - Rozmowy: {metric['conversations']}
  - Średni czas: {metric['avg_duration_min']} min
  - Resolution rate: {metric['resolution_rate']}%
  - Handoff rate: {metric['handoff_rate']}%
  - Avg retrieval score: {metric['avg_retrieval_score']}
  - Avg CSAT: {metric['avg_csat']}
"""
            
            # Add escalation reasons
            if analytics_data.get('escalation_analysis', {}).get('reasons'):
                analytics_context += f"\nPRZYCZYNY ESKALACJI:\n"
                for reason, count in sorted(analytics_data['escalation_analysis']['reasons'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    analytics_context += f"- {reason}: {count} przypadków\n"
            
            # Construct prompt with system instructions and data
            full_prompt = f"""{AGENT3_SYSTEM_PROMPT}

{analytics_context}

ZAPYTANIE UŻYTKOWNIKA:
{query}

Przeanalizuj powyższe dane i wygeneruj odpowiedź zgodnie z formatem określonym w instrukcjach systemowych.
Jeśli dane są niepełne lub brakuje kolekcji, wyraźnie to zaznacz w podsumowaniu."""

            try:
                response = llm.invoke(full_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                answer = f"{answer}\n\n---\nOdpowiedź wygenerowana przez {CHAT_MODEL} na podstawie rzeczywistych danych"
            except Exception as e:
                answer = f"{analytics_context}\n\n⚠️ LLM niedostępny: {str(e)}"
            
            total_conversations = analytics_data.get('data_availability', {}).get('conversations', 0)
            total_turns = 0
            total_documents = 0
        
        else:
            # Standard statistics query (backward compatibility)
            conv_stats = analyze_conversations("")
            turn_stats = analyze_turns("")
            kb_stats = analyze_knowledge_base("")
            
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
            
            total_conversations = conv_stats.get('total_conversations', 0)
            total_turns = turn_stats.get('total_turns', 0)
            total_documents = kb_stats.get('total_documents', 0)
        
        elapsed_time = time.time() - start_time
        answer = f"{answer}\n\nCzas odpowiedzi: {elapsed_time:.2f}s"
        
        # Save query and result to collection
        save_query_to_collection(
            query=query,
            answer=answer,
            elapsed_time=elapsed_time,
            stats_context={
                "total_conversations": total_conversations,
                "total_turns": total_turns,
                "total_documents": total_documents
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
