import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_community.chat_models import ChatOllama
import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue

logger = logging.getLogger("uvicorn")

# Add surveys module to path
SURVEYS_DIR = Path(__file__).parent / "surveys"
STATIC_DIR = Path(__file__).parent / "static"
sys.path.insert(0, str(SURVEYS_DIR))

app = FastAPI()

# Mount static files for survey HTML
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    print(f"✓ Static files mounted at /static")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(model="tinyllama", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "survey_responses")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_DIM = 128

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"✗ Failed to connect to Qdrant: {e}")
    qdrant_client = None

# Pydantic models for survey data
class SurveyResponse(BaseModel):
    response_id: str
    timestamp: str
    student_id: str
    student_email: Optional[str] = None
    album_number: Optional[str] = None
    q1_goal: Optional[str] = None
    q2_duration: Optional[str] = None
    q3_understanding: Optional[str] = None
    q4_clarity: Optional[str] = None
    q5_accuracy: Optional[str] = None
    q6_usefulness: Optional[str] = None
    q7_advanced_questions: Optional[str] = None
    q8_overall_experience: Optional[str] = None
    q9_ease_of_use: Optional[str] = None
    q10_response_time: Optional[str] = None
    q11_ai_understanding: Optional[str] = None
    q12_chat_vs_teacher: Optional[str] = None
    q13_future_use: Optional[str] = None
    q14_positive: Optional[str] = None
    q15_improvements: Optional[str] = None
    q16_problems: Optional[str] = None
    q17_additional_info: Optional[str] = None
    q18_contact_allowed: Optional[str] = None
    satisfaction_score: float

def generate_simple_embedding(text: str) -> List[float]:
    """Generate a simple hash-based embedding for text."""
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(EMBEDDING_DIM):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 127.5) - 1.0)
    
    return embedding

def prepare_survey_text(survey: Dict[str, Any]) -> str:
    """Prepare survey data as text for embedding."""
    text_parts = [
        f"Cel: {survey.get('q1_goal', 'N/A')}",
        f"Czas: {survey.get('q2_duration', 'N/A')}",
        f"Zrozumienie: {survey.get('q3_understanding', 'N/A')}",
        f"Jasność: {survey.get('q4_clarity', 'N/A')}",
        f"Dokładność: {survey.get('q5_accuracy', 'N/A')}",
        f"Użyteczność: {survey.get('q6_usefulness', 'N/A')}",
        f"Zaawansowane: {survey.get('q7_advanced_questions', 'N/A')}",
        f"Doświadczenie: {survey.get('q8_overall_experience', 'N/A')}",
        f"Łatwość: {survey.get('q9_ease_of_use', 'N/A')}",
        f"Szybkość: {survey.get('q10_response_time', 'N/A')}",
        f"AI: {survey.get('q11_ai_understanding', 'N/A')}",
        f"Chat vs nauczyciel: {survey.get('q12_chat_vs_teacher', 'N/A')}",
        f"Przyszłość: {survey.get('q13_future_use', 'N/A')}",
        f"Opinia: {survey.get('q14_positive', 'N/A')}",
        f"Ulepszenia: {survey.get('q15_improvements', 'N/A')}",
    ]
    return " ".join(text_parts)

@app.post("/api/surveys")
async def add_survey(survey: SurveyResponse) -> dict:
    """Add a new survey response to Qdrant and save to file."""
    if not qdrant_client:
        return {
            "status": "error",
            "message": "Qdrant not available",
            "response_id": survey.response_id
        }
    
    try:
        # Generate next ID (count existing points + 1)
        try:
            info = qdrant_client.get_collection(COLLECTION)
            point_id = info.points_count + 1
        except:
            point_id = 1
        
        # Prepare survey data
        survey_dict = survey.dict()
        survey_text = prepare_survey_text(survey_dict)
        embedding = generate_simple_embedding(survey_text)
        
        # Prepare payload (filter out None values)
        payload = {k: v for k, v in survey_dict.items() if v is not None}
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        
        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION,
            points=[point]
        )
        
        return {
            "status": "success",
            "message": f"Survey saved successfully",
            "response_id": survey.response_id,
            "point_id": point_id,
            "file_saved": False,
            "file_path": "Qdrant only"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "response_id": survey.response_id
        }

@app.delete("/api/surveys/range/{start_id}/{end_id}")
async def delete_survey_range(start_id: int, end_id: int) -> dict:
    """Delete a range of surveys by point_id."""
    if not qdrant_client:
        return {
            "status": "error",
            "message": "Qdrant not available"
        }
    
    try:
        deleted_count = 0
        errors = []
        
        for point_id in range(start_id, end_id + 1):
            try:
                qdrant_client.delete(
                    collection_name=COLLECTION,
                    points_selector=[point_id]
                )
                deleted_count += 1
            except Exception as e:
                errors.append(f"Point {point_id}: {str(e)}")
        
        return {
            "status": "success",
            "deleted": deleted_count,
            "errors": errors[:10] if errors else []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/run")
async def run(payload: dict):
    data = payload.get("input", "")
    result = llm.invoke(f"Analyze this data and provide insights: {data}")
    return {"insights": result.content, "collection": COLLECTION}

@app.get("/")
async def root():
    """Redirect to survey form."""
    return {"message": "Analytics Agent 3", "survey_url": "/static/survey.html"}

@app.get("/surveys")
async def survey_form():
    """Serve survey HTML form."""
    survey_path = STATIC_DIR / "survey.html"
    if survey_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(survey_path, media_type="text/html")
    return {"error": "Survey form not found"}

@app.get("/surveys/stats")
async def surveys_stats():
    """Get survey collection statistics."""
    if not qdrant_client:
        return {"error": "Qdrant not available"}
    
    try:
        info = qdrant_client.get_collection(COLLECTION)
        return {
            "collection": COLLECTION,
            "points_count": info.points_count,
            "status": "ready"
        }
    except Exception as e:
        return {"error": str(e), "collection": COLLECTION}

# OpenAI-compatible chat endpoint for Open WebUI integration
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "agent3_analytics"
    messages: List[ChatMessage]
    temperature: float = 0.7
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat endpoint for analytics agent."""
    try:
        # Get last user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        print(f"=== RECEIVED MESSAGE: '{user_message}' ===", flush=True)
        
        if not user_message:
            return {"error": "No user message found"}
        
        # Build context from conversation history
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[-5:]])
        
        # Check if query is about surveys/analytics
        if any(keyword in user_message.lower() for keyword in ['ankiet', 'survey', 'statystyk', 'ocen', 'student', 'badani', 'odpowied', 'ile', 'jaki', 'średni', 'how many', 'count', 'dni', 'okres', 'przedział', 'najniżs', 'najwyżs', 'najgorsz', 'najleps', 'pokaż', 'pokaz', 'wyświetl', 'zobacz', 'gen_resp', 'list', 'bliski', 'blisk']):
            user_lower = user_message.lower()
            
            response_text = None  # Initialize
            
            # Check for "near average" queries first
            if any(word in user_lower for word in ['bliski', 'blisk', 'średni', 'przeciętn']) and any(word in user_lower for word in ['list', 'pokaż', 'pokaz', 'wyświetl']):
                response_text = await find_surveys_near_average(tolerance=0.3, limit=10)
            else:
                # Check for extreme value queries FIRST (before specific survey ID)
                if any(word in user_lower for word in ['najniż', 'najgorsz', 'najsłab', 'najgorsze', 'najsłabsze']):
                    # Check if asking for list (10, lista, listę, podaj, pokaż, etc)
                    if any(word in user_lower for word in ['10', 'dziesięć', 'list', 'podaj', 'pokaż', 'pokaz', 'wypisz', 'wyświetl']):
                        response_text = await find_lowest_surveys(10)
                    else:
                        # Query for single lowest score
                        response_text = await find_extreme_survey(find_max=False)
                elif any(word in user_lower for word in ['najwyż', 'najlep', 'najlepsze']):
                    # Check if asking for list (10, lista, listę, podaj, pokaż, etc)
                    if any(word in user_lower for word in ['10', 'dziesięć', 'list', 'podaj', 'pokaż', 'pokaz', 'wypisz', 'wyświetl']):
                        response_text = await find_highest_surveys(10)
                    else:
                        # Query for single highest score
                        response_text = await find_extreme_survey(find_max=True)
                else:
                    # Check for specific survey ID query
                    import re
                    survey_id_pattern = r'gen_resp_\d+'
                    survey_id_matches = re.findall(survey_id_pattern, user_message)
                    
                    print(f"DEBUG: user_message = '{user_message}'", flush=True)
                    print(f"DEBUG: survey_id_matches = {survey_id_matches}", flush=True)
                    
                    # Also check for numeric ID after "ankiet"
                    if not survey_id_matches and 'ankiet' in user_lower:
                        numeric_pattern = r'\b(\d{1,3})\b'
                        numeric_matches = re.findall(numeric_pattern, user_message)
                        if numeric_matches:
                            survey_id_matches = numeric_matches
                    
                    print(f"DEBUG: final survey_id_matches = {survey_id_matches}", flush=True)
                    
                    if survey_id_matches:
                        # Specific survey query - fetch from Qdrant
                        survey_id = survey_id_matches[-1]
                        try:
                            point = None
                            if survey_id.isdigit():
                                points = qdrant_client.retrieve(
                                    collection_name=COLLECTION,
                                    ids=[int(survey_id)],
                                    with_payload=True,
                                    with_vectors=False
                                )
                                if points:
                                    point = points[0]
                            else:
                                from qdrant_client.models import Filter, FieldCondition, MatchValue
                                points, _ = qdrant_client.scroll(
                                    collection_name=COLLECTION,
                                    limit=1,
                                    with_payload=True,
                                    with_vectors=False,
                                    scroll_filter=Filter(
                                        must=[
                                            FieldCondition(
                                                key="response_id",
                                                match=MatchValue(value=survey_id)
                                            )
                                        ]
                                    )
                                )
                                if points:
                                    point = points[0]
                            
                            if point:
                                p = point.payload
                                response_text = f"""Ankieta {p.get('response_id', 'N/A')}
Data: {p.get('timestamp', 'N/A')}
Student ID: {p.get('student_id', 'N/A')}
Email: {p.get('student_email', 'N/A')}
Album: {p.get('album_number', 'N/A')}
Ocena zadowolenia: {p.get('satisfaction_score', 'N/A')}/5.0

Cel wizyty: {p.get('q1_goal', 'N/A')}
Czas trwania: {p.get('q2_duration', 'N/A')}
Zrozumienie (1-5): {p.get('q3_understanding', 'N/A')}
Jasność (1-5): {p.get('q4_clarity', 'N/A')}
Dokładność (1-5): {p.get('q5_accuracy', 'N/A')}
Przydatność (1-5): {p.get('q6_usefulness', 'N/A')}
Zaawansowane pytania (1-5): {p.get('q7_advanced_questions', 'N/A')}
Ogólne doświadczenie (1-5): {p.get('q8_overall_experience', 'N/A')}
Łatwość użycia (1-5): {p.get('q9_ease_of_use', 'N/A')}
Czas odpowiedzi (1-5): {p.get('q10_response_time', 'N/A')}
Zrozumienie AI (1-5): {p.get('q11_ai_understanding', 'N/A')}
Chat vs. nauczyciel: {p.get('q12_chat_vs_teacher', 'N/A')}
Przyszłe użycie: {p.get('q13_future_use', 'N/A')}
Pozytywne: {p.get('q14_positive', 'N/A')}
Usprawnienia: {p.get('q15_improvements', 'N/A')}
Problemy: {p.get('q16_problems', 'N/A')}
Dodatkowe info: {p.get('q17_additional_info', 'N/A')}
Kontakt dozwolony: {p.get('q18_contact_allowed', 'N/A')}"""
                            else:
                                response_text = f"Nie znaleziono ankiety: {survey_id}"
                        except Exception as e:
                            response_text = f"Błąd podczas pobierania ankiety: {str(e)}"
                    elif not response_text:
                        # Check for date range query and score range query
                        import re
                        date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
                        dates = re.findall(date_pattern, user_message)
                        
                        # Check for score range query
                        score_pattern = r'(\d+(?:\.\d+)?)\s*[-–do]+\s*(\d+(?:\.\d+)?)'
                        score_matches = re.findall(score_pattern, user_message)
                        
                        if score_matches and ('ocen' in user_lower or 'ocena' in user_lower or 'zadowolenia' in user_lower):
                            # Score range query - return directly without LLM
                            score_min = float(score_matches[0][0])
                            score_max = float(score_matches[0][1])
                            count = await count_surveys_by_score_range(score_min, score_max)
                            response_text = f"Liczba ankiet z ocenami w przedziale {score_min}-{score_max}: {count}"
                        elif len(dates) >= 2 and ('do' in user_lower or 'okres' in user_lower or 'przedział' in user_lower or '-' in user_message):
                            # Date range query - return directly without LLM
                            day1, month1, year1 = dates[0]
                            day2, month2, year2 = dates[1]
                            date_from = f"{year1}-{month1.zfill(2)}-{day1.zfill(2)}T00:00:00"
                            date_to = f"{year2}-{month2.zfill(2)}-{day2.zfill(2)}T23:59:59"
                            
                            count = await count_surveys_in_date_range(date_from, date_to)
                            response_text = f"W okresie {day1}.{month1}.{year1} - {day2}.{month2}.{year2} jest {count} ankiet."
                        elif 'ile' in user_lower and 'ankiet' in user_lower:
                            # Direct count - return without LLM
                            stats = await get_survey_context()
                            response_text = stats.split('\n')[0]
                        elif ('średni' in user_lower or 'średnia' in user_lower) and not any(word in user_lower for word in ['list', 'pokaż', 'pokaz']):
                            # Direct average - return without LLM (only if not asking for list)
                            stats = await get_survey_context()
                            lines = stats.split('\n')
                            response_text = lines[1] if len(lines) > 1 else stats
                        else:
                            # General stats - return without LLM
                            response_text = await get_survey_context()
            
            # If response_text wasn't set, provide default
            if not response_text:
                response_text = await get_survey_context()
        else:
            response_text = "Jestem agentem analitycznym zajmującym się ankietami studentów. Mogę podać:\n- Liczbę ankiet\n- Statystyki ocen\n- Ankiety z konkretnego okresu\n- Ankiety z określonym zakresem ocen\n- Najlepsze i najgorsze ankiety"
        
        # Format as OpenAI response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split())
            }
        }
    except Exception as e:
        return {"error": str(e)}

async def get_survey_context():
    """Get survey statistics for context."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        info = qdrant_client.get_collection(COLLECTION)
        total = info.points_count
        
        if total == 0:
            return "Baza danych jest pusta. Brak ankiet do analizy."
        
        # Get all surveys
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        # Calculate statistics
        scores = []
        goals = []
        durations = []
        
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            scores.append(score)
            
            if payload.get('q1_goal'):
                goals.append(payload['q1_goal'])
            if payload.get('q2_duration'):
                durations.append(payload['q2_duration'])
        
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Count goals
        from collections import Counter
        goal_counts = Counter(goals)
        top_goals = goal_counts.most_common(3)
        
        stats = f"""Liczba ankiet: {total}
Średnia ocena zadowolenia: {avg_score:.2f}/5.0
Najniższa ocena: {min_score:.2f}
Najwyższa ocena: {max_score:.2f}

Najczęstsze cele wizyt:"""
        
        for goal, count in top_goals:
            stats += f"\n- {goal}: {count} ankiet"
        
        return stats
        
    except Exception as e:
        return f"Błąd pobierania statystyk: {e}"

async def count_surveys_in_date_range(date_from: str, date_to: str):
    """Count surveys in specific date range."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        # Filter by date range
        count = 0
        for point in all_points:
            payload = point.payload or {}
            timestamp = payload.get('timestamp', '')
            if timestamp >= date_from and timestamp <= date_to:
                count += 1
        
        return count
    except Exception as e:
        return f"Błąd: {e}"

async def count_surveys_by_score_range(score_min: float, score_max: float):
    """Count surveys with satisfaction scores in specific range."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        # Filter by score range
        count = 0
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            if score >= score_min and score <= score_max:
                count += 1
        
        return count
    except Exception as e:
        return f"Błąd: {e}"

async def find_surveys_near_average(tolerance: float = 0.3, limit: int = 10):
    """Find surveys with scores close to average."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        # Get all points to calculate average
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        if not all_points:
            return "Brak ankiet w bazie."
        
        # Calculate average
        total_score = 0
        count = 0
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            if score > 0:
                total_score += score
                count += 1
        
        if count == 0:
            return "Brak ankiet z oceną."
        
        avg_score = total_score / count
        
        # Find surveys near average
        near_avg = []
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            if abs(score - avg_score) <= tolerance:
                near_avg.append({
                    'response_id': payload.get('response_id', 'N/A'),
                    'score': score,
                    'timestamp': payload.get('timestamp', 'N/A'),
                    'diff': abs(score - avg_score)
                })
        
        # Sort by difference from average
        near_avg.sort(key=lambda x: x['diff'])
        near_avg = near_avg[:limit]
        
        result = f"Średnia ocena: {avg_score:.2f}/5.0\n"
        result += f"Znaleziono {len(near_avg)} ankiet w zakresie ±{tolerance}:\n\n"
        for survey in near_avg:
            result += f"- {survey['response_id']}: {survey['score']:.2f}/5.0 (różnica: {survey['diff']:.2f})\n"
        
        return result
    except Exception as e:
        return f"Błąd: {e}"

async def find_lowest_surveys(limit: int = 10):
    """Find surveys with lowest satisfaction scores."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        if not all_points:
            return "Brak ankiet w bazie."
        
        # Create list with scores
        surveys_with_scores = []
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            surveys_with_scores.append({
                'response_id': payload.get('response_id', str(point.id)),
                'score': score,
                'timestamp': payload.get('timestamp', 'N/A')[:10],
                'goal': payload.get('q1_goal', 'N/A'),
                'student_id': payload.get('student_id', 'N/A')
            })
        
        # Sort by score (ascending) and take first 'limit'
        surveys_with_scores.sort(key=lambda x: x['score'])
        lowest = surveys_with_scores[:limit]
        
        result = f"🔻 {len(lowest)} ankiet z najniższymi ocenami:\n\n"
        for i, survey in enumerate(lowest, 1):
            result += f"{i}. {survey['response_id']}: ⭐ {survey['score']:.2f}/5.0\n"
            result += f"   Data: {survey['timestamp']}, Cel: {survey['goal']}\n\n"
        
        return result
    except Exception as e:
        return f"Błąd: {e}"

async def find_highest_surveys(limit: int = 10):
    """Find surveys with highest satisfaction scores."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        if not all_points:
            return "Brak ankiet w bazie."
        
        # Create list with scores
        surveys_with_scores = []
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            surveys_with_scores.append({
                'response_id': payload.get('response_id', str(point.id)),
                'score': score,
                'timestamp': payload.get('timestamp', 'N/A')[:10],
                'goal': payload.get('q1_goal', 'N/A'),
                'student_id': payload.get('student_id', 'N/A')
            })
        
        # Sort by score (descending) and take first 'limit'
        surveys_with_scores.sort(key=lambda x: x['score'], reverse=True)
        highest = surveys_with_scores[:limit]
        
        result = f"🔺 {len(highest)} ankiet z najwyższymi ocenami:\n\n"
        for i, survey in enumerate(highest, 1):
            result += f"{i}. {survey['response_id']}: ⭐ {survey['score']:.2f}/5.0\n"
            result += f"   Data: {survey['timestamp']}, Cel: {survey['goal']}\n\n"
        
        return result
    except Exception as e:
        return f"Błąd: {e}"

async def find_extreme_survey(find_max: bool = False):
    """Find survey with lowest or highest satisfaction score."""
    if not qdrant_client:
        return "Brak połączenia z bazą danych."
    
    try:
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
        
        if not all_points:
            return "Brak ankiet w bazie."
        
        # Find extreme
        extreme_point = None
        extreme_score = float('inf') if not find_max else float('-inf')
        
        for point in all_points:
            payload = point.payload or {}
            score = float(payload.get('satisfaction_score', 0) or 0)
            
            if find_max:
                if score > extreme_score:
                    extreme_score = score
                    extreme_point = point
            else:
                if score < extreme_score:
                    extreme_score = score
                    extreme_point = point
        
        if extreme_point:
            payload = extreme_point.payload or {}
            result = f"""{'Najwyższa' if find_max else 'Najniższa'} ocena: {extreme_score:.2f}/5.0
ID ankiety: {payload.get('response_id', extreme_point.id)}
Student: {payload.get('student_id', 'N/A')}
Data: {payload.get('timestamp', 'N/A')[:10]}
Cel: {payload.get('q1_goal', 'N/A')}"""
            return result
        
        return "Nie znaleziono ankiety."
    except Exception as e:
        return f"Błąd: {e}"

@app.get("/v1/models")
async def list_models():
    """List available models for OpenAI compatibility."""
    return {
        "object": "list",
        "data": [{
            "id": "agent3_analytics",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "agent3"
        }]
    }

@app.get("/surveys/list")
async def list_surveys(limit: int = None, date_from: str = None, date_to: str = None, score_min: float = None, score_max: float = None, album_number: str = None):
    """List survey responses with optional filtering (from Qdrant)."""
    if not qdrant_client:
        return {"error": "Qdrant not available"}
    try:
        all_points = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break

        total_count = len(all_points)

        filtered_surveys = []
        for point in all_points:
            payload = point.payload or {}
            timestamp = payload.get('timestamp', '')
            score = float(payload.get('satisfaction_score', 0) or 0)
            response_id = payload.get('response_id') or str(point.id)
            survey_album = payload.get('album_number', '')

            # Check album number filter
            if album_number and survey_album != album_number:
                continue

            # Check date filters
            if date_from and timestamp < date_from:
                continue
            if date_to and timestamp > date_to:
                continue

            # Check score filters
            if score_min is not None and score < score_min:
                continue
            if score_max is not None and score > score_max:
                continue

            filtered_surveys.append({
                'id': str(point.id),
                'response_id': response_id,
                'timestamp': timestamp,
                'satisfaction_score': score
            })

        # Sort by timestamp desc
        filtered_surveys.sort(key=lambda x: x.get('timestamp') or '', reverse=True)

        # Apply limit
        if limit:
            filtered_surveys = filtered_surveys[:limit]

        return {
            "status": "success",
            "count": len(filtered_surveys),
            "total_count": total_count,
            "surveys": filtered_surveys
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/surveys/view/{survey_id}")
async def view_survey(survey_id: str):
    """View a specific survey response (from Qdrant)."""
    if not qdrant_client:
        return {"error": "Qdrant not available"}
    try:
        point = None

        if survey_id.isdigit():
            points = qdrant_client.retrieve(
                collection_name=COLLECTION,
                ids=[int(survey_id)],
                with_payload=True,
                with_vectors=False
            )
            if points:
                point = points[0]
        else:
            points, _ = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=1,
                with_payload=True,
                with_vectors=False,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="response_id",
                            match=MatchValue(value=survey_id)
                        )
                    ]
                )
            )
            if points:
                point = points[0]

        if not point:
            return {"error": f"Survey not found: {survey_id}"}

        payload = point.payload or {}
        payload["point_id"] = point.id

        return {
            "status": "success",
            "survey_id": str(point.id),
            "data": payload
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/surveys/browse")
async def browse_surveys():
    """HTML page to browse surveys with filters."""
    from fastapi.responses import FileResponse
    return FileResponse('static/browse.html')
