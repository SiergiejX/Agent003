import os
import hashlib
import time
import uuid
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from qdrant_client import QdrantClient
from typing import List

app = FastAPI()

llm = ChatOllama(
    model="llama3",
    base_url="http://ollama:11434"
)

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
KNOWLEDGE_BASE_COLLECTION = "BazaWiedzy"
CONVERSATIONS_COLLECTION = "agent1_conversations"  # Full conversations with history
TOPICS_COLLECTION = "agent1_turns"  # Conversation topics

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# In-memory conversation storage (in production use Redis/database)
conversations = {}


def generate_simple_embedding(text: str, dim: int = 768) -> List[float]:
    """Generate simple hash-based embedding."""
    hash_obj = hashlib.sha256(text.lower().encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 255.0) * 2 - 1)
    
    return embedding


# Minimum similarity threshold - documents below this score are considered irrelevant
MIN_SIMILARITY_THRESHOLD = 0.5

def search_knowledge_base(query: str, limit: int = 3) -> List[dict]:
    """Search knowledge base for relevant documents."""
    try:
        query_embedding = generate_simple_embedding(query, dim=768)
        
        results = client.query_points(
            collection_name=KNOWLEDGE_BASE_COLLECTION,
            query=query_embedding,
            limit=limit,
            with_payload=True
        ).points
        
        documents = []
        for result in results:
            # Only include documents above similarity threshold
            if result.score >= MIN_SIMILARITY_THRESHOLD:
                documents.append({
                    "score": result.score,
                    "filename": result.payload.get("filename", ""),
                    "category": result.payload.get("category", ""),
                    "content": result.payload.get("full_content", "")
                })
        
        return documents
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []


def save_topic_to_qdrant(conversation_id: str, topic: str, first_question: str):
    """Save conversation topic to topics collection."""
    try:
        # Get next point ID
        try:
            info = client.get_collection(TOPICS_COLLECTION)
            point_id = info.points_count + 1
        except:
            point_id = 1
        
        # Generate embedding
        embedding = generate_simple_embedding(f"{topic} {first_question}", dim=768)
        
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "conversation_id": conversation_id,
                "topic": topic,
                "first_question": first_question,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "agent_id": 1
            }
        )
        
        client.upsert(collection_name=TOPICS_COLLECTION, points=[point])
        print(f"✓ Topic saved to {TOPICS_COLLECTION} (conversation_id: {conversation_id})")
    except Exception as e:
        print(f"Error saving topic: {e}")


def save_conversation_to_qdrant(conversation_id: str, conversation_data: dict):
    """Save full conversation to conversations collection."""
    try:
        # Get next point ID
        try:
            info = client.get_collection(CONVERSATIONS_COLLECTION)
            point_id = info.points_count + 1
        except:
            point_id = 1
        
        # Create embedding from all turns
        all_text = " ".join([f"{turn['question']} {turn['answer']}" for turn in conversation_data['turns']])
        embedding = generate_simple_embedding(all_text, dim=768)
        
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "conversation_id": conversation_id,
                "topic": conversation_data['topic'],
                "turns": conversation_data['turns'],
                "total_turns": len(conversation_data['turns']),
                "success": conversation_data.get('success', True),
                "start_time": conversation_data['start_time'],
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "agent_id": 1,
                "metadata": conversation_data.get('metadata', {})
            }
        )
        
        client.upsert(collection_name=CONVERSATIONS_COLLECTION, points=[point])
        print(f"✓ Conversation saved to {CONVERSATIONS_COLLECTION} (id: {conversation_id}, turns: {len(conversation_data['turns'])}, success: {conversation_data.get('success', True)})")
    except Exception as e:
        print(f"Error saving conversation: {e}")


@app.post("/start")
async def start_conversation(payload: dict):
    """Start a new conversation."""
    conversation_id = str(uuid.uuid4())
    question = payload.get("input", "")
    
    if not question:
        return {"error": "No input provided"}
    
    print(f"=== NEW CONVERSATION: {conversation_id} ===", flush=True)
    print(f"Question: '{question}'", flush=True)
    
    # Search knowledge base
    documents = search_knowledge_base(question, limit=3)
    
    if documents:
        context = "Znalezione dokumenty w bazie wiedzy:\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"[Dokument {i}: {doc['filename']} - {doc['category']} (podobieństwo: {doc['score']:.2f})]\n"
            context += f"{doc['content'][:500]}...\n\n"
        
        prompt = f"""{context}

Pytanie studenta: {question}

WAŻNE INSTRUKCJE:
- Odpowiadaj TYLKO na podstawie informacji zawartych w powyższych dokumentach
- Jeśli odpowiedź na pytanie NIE znajduje się w dokumentach, powiedz: "Przepraszam, nie mam informacji na ten temat w mojej bazie wiedzy. Skontaktuj się z dziekanatem lub sprawdź Extranet."
- NIE wymyślaj informacji, NIE domyślaj się, NIE udzielaj ogólnych odpowiedzi
- Odpowiadaj zwięźle i konkretnie"""
    else:
        # No relevant documents found - do not make up answers
        answer = "Przepraszam, nie znalazłem informacji na ten temat w mojej bazie wiedzy. Aby uzyskać pomoc, skontaktuj się z dziekanatem lub sprawdź portal Extranet uczelni."
        
        # Extract topic
        topic = question[:100] if len(question) <= 100 else question[:97] + "..."
        
        # Initialize conversation
        conversations[conversation_id] = {
            "topic": topic,
            "turns": [{"question": question, "answer": answer}],
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "success": None
        }
        
        # Save topic
        save_topic_to_qdrant(conversation_id, topic, question)
        
        print(f"⚠️ No relevant documents found (below threshold {MIN_SIMILARITY_THRESHOLD})", flush=True)
        
        return {
            "conversation_id": conversation_id,
            "result": answer,
            "turn": 1,
            "documents_found": 0
        }
    
    # Generate answer
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Przepraszam, wystąpił błąd: {e}"
        print(f"LLM error: {e}", flush=True)
    
    # Extract topic from first question
    topic = question[:100] if len(question) <= 100 else question[:97] + "..."
    
    # Initialize conversation
    conversations[conversation_id] = {
        "topic": topic,
        "turns": [{"question": question, "answer": answer}],
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": None  # Will be set when conversation ends
    }
    
    # Save topic
    save_topic_to_qdrant(conversation_id, topic, question)
    
    return {
        "conversation_id": conversation_id,
        "result": answer,
        "turn": 1,
        "documents_found": len(documents)
    }


@app.post("/continue")
async def continue_conversation(payload: dict):
    """Continue existing conversation."""
    conversation_id = payload.get("conversation_id", "")
    question = payload.get("input", "")
    
    if not conversation_id or conversation_id not in conversations:
        return {"error": "Invalid conversation_id or conversation not found"}
    
    if not question:
        return {"error": "No input provided"}
    
    print(f"=== CONTINUE CONVERSATION: {conversation_id} ===", flush=True)
    print(f"Question: '{question}'", flush=True)
    
    # Search knowledge base
    documents = search_knowledge_base(question, limit=3)
    
    # Build context with conversation history
    conv = conversations[conversation_id]
    history_context = "Historia rozmowy:\n"
    for i, turn in enumerate(conv['turns'][-3:], 1):  # Last 3 turns
        history_context += f"\nPytanie {i}: {turn['question']}\nOdpowiedź {i}: {turn['answer']}\n"
    
    if documents:
        doc_context = "\n\nZnalezione dokumenty w bazie wiedzy:\n\n"
        for i, doc in enumerate(documents, 1):
            doc_context += f"[Dokument {i}: {doc['filename']} - {doc['category']} (podobieństwo: {doc['score']:.2f})]\n"
            doc_context += f"{doc['content'][:500]}...\n\n"
        
        prompt = f"""{history_context}{doc_context}

Nowe pytanie studenta: {question}

WAŻNE INSTRUKCJE:
- Odpowiadaj TYLKO na podstawie dokumentów lub wcześniejszej rozmowy
- Jeśli odpowiedź NIE znajduje się w dokumentach ani w historii, powiedz: "Przepraszam, nie mam informacji na ten temat. Skontaktuj się z dziekanatem."
- NIE wymyślaj informacji, NIE domyślaj się
- Odpowiadaj zwięźle i konkretnie"""
    else:
        # No relevant documents - can only refer to conversation history
        answer = "Przepraszam, nie znalazłem dodatkowych informacji na ten temat w mojej bazie wiedzy. Jeśli Twoje pytanie wykracza poza tematykę naszej rozmowy, skontaktuj się z dziekanatem lub sprawdź portal Extranet."
        
        # Add turn to conversation
        conv['turns'].append({"question": question, "answer": answer})
        
        print(f"⚠️ No relevant documents found (below threshold {MIN_SIMILARITY_THRESHOLD})", flush=True)
        
        return {
            "conversation_id": conversation_id,
            "result": answer,
            "turn": len(conv['turns']),
            "documents_found": 0
        }
    
    # Generate answer
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Przepraszam, wystąpił błąd: {e}"
        print(f"LLM error: {e}", flush=True)
    
    # Add turn to conversation
    conv['turns'].append({"question": question, "answer": answer})
    
    return {
        "conversation_id": conversation_id,
        "result": answer,
        "turn": len(conv['turns']),
        "documents_found": len(documents)
    }


@app.post("/end")
async def end_conversation(payload: dict):
    """End conversation and save to Qdrant."""
    conversation_id = payload.get("conversation_id", "")
    success = payload.get("success", True)  # Did conversation end successfully?
    
    if not conversation_id or conversation_id not in conversations:
        return {"error": "Invalid conversation_id or conversation not found"}
    
    print(f"=== END CONVERSATION: {conversation_id} (success: {success}) ===", flush=True)
    
    conv = conversations[conversation_id]
    conv['success'] = success
    
    # Save full conversation to Qdrant
    save_conversation_to_qdrant(conversation_id, conv)
    
    # Remove from memory
    total_turns = len(conv['turns'])
    del conversations[conversation_id]
    
    return {
        "conversation_id": conversation_id,
        "success": success,
        "total_turns": total_turns,
        "saved": True
    }


@app.post("/run")
async def run(payload: dict):
    """
    Handle student queries using RAG (Retrieval-Augmented Generation).
    Legacy endpoint - creates single-turn conversation.
    
    For multi-turn conversations use: /start, /continue, /end
    """
    question = payload.get("input", "")
    
    if not question:
        return {"error": "No input provided"}
    
    print(f"=== STUDENT QUERY (legacy): '{question}' ===", flush=True)
    
    # Search knowledge base
    documents = search_knowledge_base(question, limit=3)
    
    if documents:
        context = "Znalezione dokumenty w bazie wiedzy:\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"[Dokument {i}: {doc['filename']} - {doc['category']} (podobieństwo: {doc['score']:.2f})]\n"
            context += f"{doc['content'][:500]}...\n\n"
        
        prompt = f"""{context}

Pytanie studenta: {question}

WAŻNE INSTRUKCJE:
- Odpowiadaj TYLKO na podstawie informacji zawartych w powyższych dokumentach
- Jeśli odpowiedź na pytanie NIE znajduje się w dokumentach, powiedz: "Przepraszam, nie mam informacji na ten temat w mojej bazie wiedzy. Skontaktuj się z dziekanatem lub sprawdź Extranet."
- NIE wymyślaj informacji, NIE domyślaj się, NIE udzielaj ogólnych odpowiedzi
- Odpowiadaj zwięźle i konkretnie"""
        
        print(f"✓ Found {len(documents)} relevant documents", flush=True)
    else:
        # No relevant documents found - return explicit message
        print(f"⚠️ No relevant documents found (below threshold {MIN_SIMILARITY_THRESHOLD})", flush=True)
        
        answer = "Przepraszam, nie znalazłem informacji na ten temat w mojej bazie wiedzy. Aby uzyskać pomoc, skontaktuj się z dziekanatem lub sprawdź portal Extranet uczelni."
        
        # Save as single-turn conversation
        conversation_id = str(uuid.uuid4())
        topic = question[:100] if len(question) <= 100 else question[:97] + "..."
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        conversation_data = {
            "topic": topic,
            "turns": [{"question": question, "answer": answer}],
            "start_time": timestamp,
            "success": True,
            "metadata": {"type": "single_turn", "via": "legacy_run_endpoint", "no_knowledge": True}
        }
        
        # Save both topic and conversation
        save_topic_to_qdrant(conversation_id, topic, question)
        save_conversation_to_qdrant(conversation_id, conversation_data)
        
        return {
            "result": answer,
            "conversation_id": conversation_id,
            "knowledge_base": KNOWLEDGE_BASE_COLLECTION,
            "documents_found": 0
        }
    
    # Generate answer
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Przepraszam, wystąpił błąd: {e}"
        print(f"LLM error: {e}", flush=True)
    
    # Create single-turn conversation and save immediately
    conversation_id = str(uuid.uuid4())
    topic = question[:100] if len(question) <= 100 else question[:97] + "..."
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    conversation_data = {
        "topic": topic,
        "turns": [{"question": question, "answer": answer}],
        "start_time": timestamp,
        "success": True,  # Assume success for single-turn
        "metadata": {"type": "single_turn", "via": "legacy_run_endpoint"}
    }
    
    # Save both topic and conversation
    save_topic_to_qdrant(conversation_id, topic, question)
    save_conversation_to_qdrant(conversation_id, conversation_data)
    
    return {
        "result": answer,
        "conversation_id": conversation_id,
        "knowledge_base": KNOWLEDGE_BASE_COLLECTION,
        "documents_found": len(documents)
    }
