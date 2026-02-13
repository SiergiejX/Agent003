import os
import hashlib
import time
import uuid
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from qdrant_client import QdrantClient
from typing import List

app = FastAPI()
llm = ChatOllama(model="llama3", base_url="http://ollama:11434")

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
TICKETS_COLLECTION = "agent2_tickets"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# In-memory ticket storage (in production use Redis/database)
tickets = {}


def generate_simple_embedding(text: str, dim: int = 768) -> List[float]:
    """Generate simple hash-based embedding."""
    hash_obj = hashlib.sha256(text.lower().encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 255.0) * 2 - 1)
    
    return embedding


def save_ticket_to_qdrant(ticket_id: str, ticket_data: dict):
    """Save ticket to Qdrant."""
    try:
        # Get next point ID
        try:
            info = client.get_collection(TICKETS_COLLECTION)
            point_id = info.points_count + 1
        except:
            point_id = 1
        
        # Generate embedding from ticket subject and description
        embedding = generate_simple_embedding(f"{ticket_data['subject']} {ticket_data.get('description', '')}", dim=768)
        
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "ticket_id": ticket_id,
                "subject": ticket_data['subject'],
                "description": ticket_data.get('description', ''),
                "category": ticket_data.get('category', 'general'),
                "priority": ticket_data.get('priority', 'normal'),
                "status": ticket_data['status'],
                "created_at": ticket_data['created_at'],
                "first_response_at": ticket_data.get('first_response_at'),
                "closed_at": ticket_data.get('closed_at'),
                "resolved": ticket_data.get('resolved', False),
                "agent_id": 2,
                "metadata": {
                    "time_to_first_response_minutes": ticket_data.get('time_to_first_response_minutes'),
                    "time_to_close_minutes": ticket_data.get('time_to_close_minutes'),
                    "resolution_notes": ticket_data.get('resolution_notes', '')
                }
            }
        )
        
        client.upsert(collection_name=TICKETS_COLLECTION, points=[point])
        print(f"✓ Ticket saved to {TICKETS_COLLECTION} (id: {ticket_id}, status: {ticket_data['status']}, resolved: {ticket_data.get('resolved', False)})")
    except Exception as e:
        print(f"Error saving ticket: {e}")


@app.post("/create")
async def create_ticket(payload: dict):
    """Create new ticket."""
    subject = payload.get("subject", "")
    description = payload.get("description", "")
    category = payload.get("category", "general")
    priority = payload.get("priority", "normal")
    
    if not subject:
        return {"error": "Subject is required"}
    
    ticket_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Classify ticket using LLM (optional)
    try:
        classification = llm.invoke(f"Classify this student ticket into category (grades/scholarships/exams/other): {subject}")
        suggested_category = classification.content.strip().lower()
        if suggested_category in ["grades", "scholarships", "exams", "other"]:
            category = suggested_category
    except Exception as e:
        print(f"LLM classification error: {e}")
    
    tickets[ticket_id] = {
        "ticket_id": ticket_id,
        "subject": subject,
        "description": description,
        "category": category,
        "priority": priority,
        "status": "new",
        "created_at": timestamp,
        "first_response_at": None,
        "closed_at": None,
        "resolved": False,
        "time_to_first_response_minutes": None,
        "time_to_close_minutes": None
    }
    
    print(f"=== NEW TICKET: {ticket_id} ===")
    print(f"Subject: {subject}")
    print(f"Category: {category}")
    
    return {
        "ticket_id": ticket_id,
        "status": "new",
        "category": category,
        "created_at": timestamp
    }


@app.post("/respond")
async def respond_ticket(payload: dict):
    """Mark ticket as responded (first response from BOS)."""
    ticket_id = payload.get("ticket_id", "")
    response_text = payload.get("response", "")
    
    if not ticket_id or ticket_id not in tickets:
        return {"error": "Invalid ticket_id or ticket not found"}
    
    ticket = tickets[ticket_id]
    
    if ticket['status'] == 'new':
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ticket['first_response_at'] = timestamp
        ticket['status'] = 'in_progress'
        
        # Calculate time to first response (simple approximation - minutes since creation)
        # In production, use proper datetime parsing
        ticket['time_to_first_response_minutes'] = 15  # Placeholder
        
        print(f"=== FIRST RESPONSE: {ticket_id} ===")
        print(f"Time to response: {ticket['time_to_first_response_minutes']} minutes")
    
    return {
        "ticket_id": ticket_id,
        "status": ticket['status'],
        "first_response_at": ticket['first_response_at']
    }


@app.post("/close")
async def close_ticket(payload: dict):
    """Close ticket and save to Qdrant."""
    ticket_id = payload.get("ticket_id", "")
    resolved = payload.get("resolved", True)  # Was issue actually resolved?
    resolution_notes = payload.get("notes", "")
    
    if not ticket_id or ticket_id not in tickets:
        return {"error": "Invalid ticket_id or ticket not found"}
    
    ticket = tickets[ticket_id]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    ticket['status'] = 'closed'
    ticket['closed_at'] = timestamp
    ticket['resolved'] = resolved
    ticket['resolution_notes'] = resolution_notes
    
    # Calculate time to close (placeholder)
    ticket['time_to_close_minutes'] = 120  # Placeholder
    
    print(f"=== CLOSE TICKET: {ticket_id} ===")
    print(f"Resolved: {resolved}")
    print(f"Time to close: {ticket['time_to_close_minutes']} minutes")
    
    # Save to Qdrant
    save_ticket_to_qdrant(ticket_id, ticket)
    
    # Remove from memory
    del tickets[ticket_id]
    
    return {
        "ticket_id": ticket_id,
        "status": "closed",
        "resolved": resolved,
        "closed_at": timestamp,
        "time_to_first_response_minutes": ticket.get('time_to_first_response_minutes'),
        "time_to_close_minutes": ticket['time_to_close_minutes']
    }


@app.post("/run")
async def run(payload: dict):
    """
    Legacy endpoint for backward compatibility.
    Creates and immediately closes a ticket.
    """
    subject = payload.get("input", "")
    
    if not subject:
        return {"error": "No input provided"}
    
    # Create ticket
    ticket_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Classify using LLM
    try:
        response = llm.invoke(f"Classify this student ticket: {subject}")
        classification = response.content
    except Exception as e:
        classification = f"Error: {e}"
    
    # Create and save ticket immediately (single-operation ticket)
    ticket_data = {
        "subject": subject,
        "description": "",
        "category": "general",
        "priority": "normal",
        "status": "closed",
        "created_at": timestamp,
        "first_response_at": timestamp,
        "closed_at": timestamp,
        "resolved": True,
        "time_to_first_response_minutes": 5,
        "time_to_close_minutes": 10,
        "resolution_notes": f"Auto-processed: {classification}"
    }
    
    save_ticket_to_qdrant(ticket_id, ticket_data)
    
    return {
        "ticket_id": ticket_id,
        "ticket_classification": classification,
        "collection": TICKETS_COLLECTION
    }
