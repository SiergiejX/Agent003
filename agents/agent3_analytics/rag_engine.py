"""
RAG Engine for Agent 3 Analytics
Retrieval Augmented Generation using Qdrant vector search
"""

from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import urllib.request
import json

class RAGEngine:
    """
    RAG Engine that performs semantic search across multiple Qdrant collections
    and returns relevant context for LLM prompting.
    """
    
    def __init__(self, client: QdrantClient, ollama_base_url: str, embedding_model: str):
        self.client = client
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        
        # Extract Qdrant URL from client
        try:
            self.qdrant_url = f"http://{client._client._host}:{client._client._port}"
        except:
            # Fallback
            self.qdrant_url = "http://qdrant:6333"
        
        # Collection mappings with descriptions for better routing
        self.collections_config = {
            # Academic analytics collections
            "a3_student_profiles_snapshot": {
                "description": "Snapshoty profili studentów - segmentacja i porównania",
                "keywords": ["student", "profil", "gpa", "stypendium", "rok studiów", "kierunek", "wydział"],
                "weight": 1.0
            },
            "a3_academic_events": {
                "description": "Zdarzenia akademickie - skreślenia, urlopy, ukończenia",
                "keywords": ["skreśleni", "urlop", "ukończeni", "enrollment", "dropout", "graduation", "absolwent", "zdarzeni"],
                "weight": 1.0
            },
            "a3_course_performance": {
                "description": "Wyniki studentów na poziomie przedmiotów",
                "keywords": ["przedmiot", "kurs", "ocena", "zaliczeni", "egzamin", "niezaliczeni"],
                "weight": 0.9
            },
            "a3_retention_cohorts": {
                "description": "Kohorty i wskaźniki retencji studentów",
                "keywords": ["retencja", "kohorta", "rocznik", "wskaźnik ukończen", "utrzymani"],
                "weight": 1.0
            },
            "a3_support_interactions_summary": {
                "description": "Podsumowania zgłoszeń studentów z systemu ticketowego",
                "keywords": ["zgłoszeni", "ticket", "problem", "wsparcie", "pomoc", "eskalacj"],
                "weight": 0.9
            },
            "a3_policies_and_rules_analytics": {
                "description": "Regulaminy i procedury akademickie",
                "keywords": ["regulamin", "procedura", "zasady", "przepisy", "zarządzeni"],
                "weight": 0.7
            },
            "a3_reports_and_insights": {
                "description": "Raporty i analizy wygenerowane przez Agent3",
                "keywords": ["raport", "analiza", "insight", "rekomendacja", "wnioski"],
                "weight": 0.8
            },
            "a3_metrics_timeseries": {
                "description": "Szeregi czasowe metryk akademickich",
                "keywords": ["trend", "zmiana", "czas", "miesięczn", "roczn", "wykres"],
                "weight": 0.9
            },
            "a3_analytics_queries_history": {
                "description": "Historia zapytań analitycznych z wynikami",
                "keywords": ["zapytanie", "pytanie", "query", "historia analiz"],
                "weight": 0.6
            }
        }
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama."""
        print(f"[RAG] generate_embedding() called for text: {text[:80]}...")
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text[:1000]  # Limit text length
            }
            print(f"[RAG] Calling Ollama API at {self.ollama_base_url}/api/embeddings...")
            request = urllib.request.Request(
                url=f"{self.ollama_base_url}/api/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            print(f"[RAG] Waiting for Ollama response (timeout: 60s)...")
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            print(f"[RAG] Ollama responded successfully")
            embedding = result.get("embedding", [])
            if not embedding:
                print(f"⚠️  Warning: Empty embedding for: {text[:50]}...")
                return [0.0] * 768  # Fallback
            print(f"[RAG] Embedding generated successfully, dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"❌ ERROR generating embedding: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return [0.0] * 768
    
    def select_collections(self, query: str, top_k: int = 5) -> List[str]:
        """
        Select most relevant collections based on query keywords.
        Returns list of collection names sorted by relevance.
        """
        query_lower = query.lower()
        scores = {}
        
        for collection_name, config in self.collections_config.items():
            score = 0.0
            
            # Check keyword matches
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1.0
            
            # Apply collection weight
            score *= config["weight"]
            
            if score > 0:
                scores[collection_name] = score
        
        # If no keyword matches, use all major collections
        if not scores:
            return [
                "a3_student_profiles_snapshot",
                "a3_academic_events",
                "a3_retention_cohorts",
                "a3_support_interactions_summary",
                "a3_course_performance"
            ]
        
        # Sort by score and return top_k
        sorted_collections = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [coll for coll, score in sorted_collections[:top_k]]
        
        print(f"[RAG] Selected collections: {selected}")
        return selected
    
    def search_collection(
        self, 
        collection_name: str, 
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.2,  # Lowered from 0.3
        time_range_days: Optional[int] = 7
    ) -> List[Dict[str, Any]]:
        """
        Search single collection with semantic similarity.
        Returns list of relevant documents with scores.
        """
        try:
            # Check if collection has timestamp field by sampling
            sample = self.client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True
            )[0]
            
            has_timestamp = any('timestamp' in p.payload for p in sample) if sample else False
            
            # Build filter for time range only if timestamps exist
            search_filter = None
            if time_range_days and has_timestamp:
                cutoff_date = (datetime.now() - timedelta(days=time_range_days)).isoformat()
                search_filter = {
                    "must": [
                        {
                            "key": "timestamp",
                            "range": {
                                "gte": cutoff_date
                            }
                        }
                    ]
                }
                print(f"[RAG] {collection_name}: Using time filter (last {time_range_days} days)")
            else:
                if time_range_days and not has_timestamp:
                    print(f"[RAG] {collection_name}: No timestamp field - searching all data")
            
            # Perform search using Qdrant REST API (stable across versions)
            import requests
            search_url = f"{self.qdrant_url}/collections/{collection_name}/points/search"
            
            search_payload = {
                "vector": query_vector,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vector": False
            }
            
            if search_filter:
                search_payload["filter"] = search_filter
            
            response = requests.post(search_url, json=search_payload, timeout=30)
            response.raise_for_status()
            result_data = response.json()
            search_result = result_data.get("result", [])
            
            # Format results from HTTP API
            results = []
            for hit in search_result:
                results.append({
                    "collection": collection_name,
                    "score": hit.get("score", 0),
                    "id": hit.get("id"),
                    "payload": hit.get("payload", {})
                })
            
            print(f"[RAG] {collection_name}: found {len(results)} results (score >= {score_threshold})")
            return results
            
        except Exception as e:
            print(f"❌ ERROR searching {collection_name}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return []
    
    def multi_collection_search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit_per_collection: int = 10,
        score_threshold: float = 0.2,  # Lowered from 0.3
        time_range_days: int = 7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search multiple collections and aggregate results.
        Returns dict with collection names as keys and results as values.
        """
        print(f"[RAG] multi_collection_search() called")
        print(f"[RAG] - Query: {query[:100]}...")
        print(f"[RAG] - Collections: {collections}")
        print(f"[RAG] - Limit per collection: {limit_per_collection}")
        print(f"[RAG] - Score threshold: {score_threshold}")
        
        # Generate query embedding
        print(f"[RAG] Generating embedding for query: {query[:100]}...")
        query_vector = self.generate_embedding(query)
        print(f"[RAG] Embedding generated, dimension: {len(query_vector) if query_vector else 'None'}")
        
        # Auto-select collections if not provided
        if not collections:
            print(f"[RAG] Auto-selecting collections...")
            collections = self.select_collections(query, top_k=5)
            print(f"[RAG] Auto-selected: {collections}")
        
        # Search each collection
        all_results = {}
        for collection_name in collections:
            results = self.search_collection(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit_per_collection,
                score_threshold=score_threshold,
                time_range_days=time_range_days
            )
            if results:
                all_results[collection_name] = results
        
        return all_results
    
    def format_context_for_llm(
        self,
        search_results: Dict[str, List[Dict[str, Any]]],
        max_results_per_collection: int = 2,  # Reduced from 3
        max_total_tokens: int = 1200  # Reduced from 2000 to fit in 4096 context with prompt overhead
    ) -> str:
        """
        Format search results into context string for LLM.
        Includes most relevant results while respecting token limits.
        """
        context_parts = []
        total_tokens = 0
        
        # Sort collections by total relevance (sum of scores)
        collection_scores = {}
        for coll, results in search_results.items():
            collection_scores[coll] = sum(r["score"] for r in results)
        
        sorted_collections = sorted(
            collection_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        context_parts.append("=== ZNALEZIONE DANE (RAG RETRIEVAL) ===\n")
        
        for collection_name, _ in sorted_collections:
            results = search_results[collection_name]
            
            # Collection header
            config = self.collections_config.get(collection_name, {})
            description = config.get("description", collection_name)
            context_parts.append(f"\n## Źródło: {collection_name}")
            context_parts.append(f"Opis: {description}")
            context_parts.append(f"Znaleziono: {len(results)} rekordów\n")
            
            # Add top results
            for idx, result in enumerate(results[:max_results_per_collection], 1):
                score = result["score"]
                payload = result["payload"]
                
                # Format payload based on collection type
                formatted = self._format_payload(collection_name, payload, score)
                
                # Estimate tokens (rough: 1 token ≈ 4 chars)
                result_tokens = len(formatted) // 4
                
                if total_tokens + result_tokens > max_total_tokens:
                    context_parts.append(f"\n[Osiągnięto limit tokenów - pominięto pozostałe wyniki]")
                    break
                
                context_parts.append(f"\n### Wynik #{idx} (podobieństwo: {score:.3f})")
                context_parts.append(formatted)
                
                total_tokens += result_tokens
            
            if total_tokens >= max_total_tokens:
                break
        
        context_parts.append(f"\n\n=== KONIEC DANYCH (łącznie ~{total_tokens} tokenów) ===\n")
        
        return "\n".join(context_parts)
    
    def _format_payload(self, collection_name: str, payload: Dict[str, Any], score: float) -> str:
        """Format payload based on collection type."""
        
        if collection_name == "a3_student_profiles_snapshot":
            return f"""
- Student hash: {payload.get('student_hash', 'N/A')}
- Rok akademicki: {payload.get('academic_year', 'N/A')}
- Kierunek: {payload.get('program_id', 'N/A')} (Wydział: {payload.get('faculty_id', 'N/A')})
- Rok studiów: {payload.get('year_of_study', 'N/A')}, Tryb: {payload.get('mode', 'N/A')}
- Status: {payload.get('status', 'N/A')}
- GPA: {payload.get('gpa', 'N/A')}, ECTS: {payload.get('ects_completed', 0)}/{payload.get('ects_expected', 180)}
- Niezaliczenia: {payload.get('failed_courses_count', 0)}
- Ryzyko odejścia: {payload.get('withdrawal_risk_score', 'N/A')}
- Stypendium: {'Tak' if payload.get('scholarship_flag') else 'Nie'}
"""
        
        elif collection_name == "a3_academic_events":
            return f"""
- Typ zdarzenia: {payload.get('event_type', 'UNKNOWN')}
- Data: {payload.get('event_date', 'N/A')}
- Rok akademicki: {payload.get('academic_year', 'N/A')}, Semestr: {payload.get('semester', 'N/A')}
- Kierunek: {payload.get('program_id', 'N/A')} (Wydział: {payload.get('faculty_id', 'N/A')})
- Powód: {payload.get('reason_text', payload.get('reason_code', 'Brak'))}
- Waga: {payload.get('severity', 'N/A')}
"""
        
        elif collection_name == "a3_course_performance":
            return f"""
- Przedmiot: {payload.get('course_name', 'N/A')} (ID: {payload.get('course_id', 'N/A')})
- Kierunek: {payload.get('program_id', 'N/A')} (Wydział: {payload.get('faculty_id', 'N/A')})
- Termin: {payload.get('term', 'N/A')}
- Próba: {payload.get('attempt_no', 1)}
- Ocena: {payload.get('final_grade', 'N/A')}, Zaliczony: {'Tak' if payload.get('passed') else 'Nie'}
- ECTS: {payload.get('ects', 'N/A')}
- Typ: {payload.get('exam_type', 'N/A')}, Kategoria: {payload.get('course_category', 'N/A')}
"""
        
        elif collection_name == "a3_retention_cohorts":
            retention_1_sem = payload.get('retention_1_sem', 0)
            retention_2_sem = payload.get('retention_2_sem', 0)
            retention_1_year = payload.get('retention_1_year', 0)
            graduation_rate = payload.get('graduation_rate', 0)
            
            return f"""
- Kohorta: {payload.get('cohort_id', 'N/A')}
- Rok wejścia: {payload.get('entry_year', 'N/A')}
- Kierunek: {payload.get('program_id', 'N/A')} (Wydział: {payload.get('faculty_id', 'N/A')})
- Tryb: {payload.get('mode', 'N/A')}, Stopień: {payload.get('level', 'N/A')}
- Wielkość kohorty: {payload.get('cohort_size', 'N/A')}
- Retencja (1 sem/2 sem/1 rok): {retention_1_sem*100:.1f}% / {retention_2_sem*100:.1f}% / {retention_1_year*100:.1f}%
- Wskaźnik ukończenia: {graduation_rate*100:.1f}%
- Średnie GPA: {payload.get('avg_gpa', 'N/A')}
"""
        
        elif collection_name == "a3_support_interactions_summary":
            return f"""
- Ticket hash: {payload.get('ticket_id_hash', 'N/A')}
- Data: {payload.get('created_at', 'N/A')}
- Kategoria: {payload.get('category', 'N/A')} / {payload.get('subcategory', 'N/A')}
- Sentiment: {payload.get('sentiment', 'N/A')}
- Priorytet: {payload.get('priority', 'N/A')}
- Rozwiązany: {'Tak' if payload.get('resolved') else 'Nie'}
- Czas rozwiązania: {payload.get('resolution_time_hours', 'N/A')} godz.
"""
        
        elif collection_name == "a3_policies_and_rules_analytics":
            return f"""
- Typ dokumentu: {payload.get('doc_type', 'N/A')}
- Temat: {payload.get('topic', 'N/A')}
- Obowiązuje od: {payload.get('effective_from', 'N/A')}
- Obowiązuje do: {payload.get('effective_to', 'obecnie')}
- Wersja: {payload.get('version', 'N/A')}
- Źródło: {payload.get('source', 'N/A')}
"""
        
        elif collection_name == "a3_reports_and_insights":
            return f"""
- Raport ID: {payload.get('report_id', 'N/A')}
- Data utworzenia: {payload.get('created_at', 'N/A')}
- Zakres: {payload.get('scope', 'N/A')}
- Okres: {payload.get('period_start', 'N/A')} - {payload.get('period_end', 'N/A')}
- Tagi KPI: {', '.join(payload.get('kpi_tags', []))}
- Pewność: {payload.get('confidence_score', 'N/A')}
- Źródła danych: {', '.join(payload.get('data_sources', []))}
"""
        
        elif collection_name == "a3_metrics_timeseries":
            return f"""
- Metryka: {payload.get('metric_name', 'N/A')}
- Zakres: {payload.get('scope', 'N/A')}
- Granularność: {payload.get('granularity', 'N/A')}
- Okres: {payload.get('period_from', 'N/A')} - {payload.get('period_to', 'N/A')}
- Jednostki: {payload.get('units', 'N/A')}
- Wartości: {payload.get('values', [])}
"""
        
        elif collection_name == "a3_analytics_queries_history":
            return f"""
- Zapytanie: {payload.get('query_text', 'N/A')}
- Data: {payload.get('created_at', 'N/A')}
- Intent: {payload.get('intent', 'N/A')}
- Tagi: {', '.join(payload.get('topic_tags', []))}
- Zakres: {payload.get('scope', 'N/A')}
- Streszczenie odpowiedzi: {payload.get('answer_summary', '')[:150]}...
- Pewność: {payload.get('confidence_score', 'N/A')}
"""
        
        else:
            # Generic format for unknown collections
            important_keys = ['student_hash', 'event_type', 'timestamp', 'created_at', 'context', 
                            'summary', 'description', 'value', 'count', 'program_id', 'faculty_id']
            lines = []
            for key in important_keys:
                if key in payload:
                    lines.append(f"- {key}: {payload[key]}")
            
            # Add any other interesting keys
            for key, value in payload.items():
                if key not in important_keys and not key.startswith('_'):
                    if isinstance(value, (str, int, float, bool)):
                        lines.append(f"- {key}: {value}")
            
            return "\n".join(lines) if lines else "- Brak danych do wyświetlenia"
    
    def retrieve_and_generate(
        self,
        query: str,
        system_prompt: str,
        llm_invoke_func,
        collections: Optional[List[str]] = None,
        time_range_days: int = 7,
        conversation_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieve → Augment → Generate.
        
        Args:
            query: User query
            system_prompt: System prompt for LLM
            llm_invoke_func: Function to invoke LLM (e.g., llm.invoke)
            collections: Optional list of collections to search
            time_range_days: Time range for filtering results
            conversation_history: Optional conversation history for context
        
        Returns:
            Dict with 'answer', 'context', 'sources', 'metadata'
        """
        print(f"\n{'='*60}")
        print(f"[RAG PIPELINE] Starting for query: {query[:100]}...")
        print(f"[RAG PIPELINE] DEBUG: Collections input: {collections}")
        print(f"[RAG PIPELINE] DEBUG: Time range: {time_range_days} days")
        print(f"{'='*60}")
        
        # Step 1: Retrieve
        print(f"[RAG PIPELINE] STEP 1: Starting multi_collection_search()...")
        search_results = self.multi_collection_search(
            query=query,
            collections=collections,
            limit_per_collection=10,
            score_threshold=0.3,
            time_range_days=time_range_days
        )
        print(f"[RAG PIPELINE] STEP 1: Search completed. Found {sum(len(r) for r in search_results.values())} results")
        
        if not search_results:
            print(f"[RAG PIPELINE] ⚠️ No results found, returning empty response")
            return {
                "answer": "Nie znaleziono żadnych danych pasujących do zapytania. Spróbuj przeformułować pytanie lub sprawdź czy kolekcje zawierają dane.",
                "context": "",
                "sources": [],
                "metadata": {
                    "retrieval_success": False,
                    "collections_searched": collections or [],
                    "total_results": 0
                }
            }
        
        # Step 2: Format context
        print(f"[RAG PIPELINE] STEP 2: Formatting context from {len(search_results)} collections...")
        context = self.format_context_for_llm(search_results, max_results_per_collection=5)
        print(f"[RAG PIPELINE] STEP 2: Context formatted, length: {len(context)} chars")
        
        # Step 3: Build augmented prompt with conversation history
        print(f"[RAG PIPELINE] STEP 3: Building augmented prompt...")
        
        # Add conversation history if provided
        history_context = ""
        if conversation_history and len(conversation_history) > 1:
            # Include last 2 exchanges maximum - limit to 300 chars total to prevent context overflow
            recent_history = conversation_history[-3:]  # Last 1-2 pairs
            history_text = "\n".join(recent_history)[:300]  # Max 300 chars
            history_context = f"\n\nHISTORIA (ostatnie wiadomości):\n{history_text}\n"
            print(f"[RAG PIPELINE] Including {len(recent_history)} messages from conversation history ({len(history_text)} chars)")
        
        augmented_prompt = f"""{system_prompt}

{context}
{history_context}
ZAPYTANIE UŻYTKOWNIKA:
{query}

INSTRUKCJE:
⚠️ ODPOWIADAJ WYŁĄCZNIE PO POLSKU! Nigdy nie używaj angielskiego ani innych języków.
1. Jeśli to pytanie kontynuuje poprzednią rozmowę, uwzględnij kontekst historii
2. Odpowiedz WYŁĄCZNIE na podstawie powyższych danych ze źródeł
3. Cytuj konkretne liczby i metryki z danych
4. Wskaż źródło informacji (nazwę kolekcji)
5. Jeśli danych nie ma w kontekście, wyraźnie to powiedź PO POLSKU
6. NIE wymyślaj danych - używaj tylko tego co zostało znalezione
7. Format odpowiedzi dostosuj do typu zapytania (tabela, lista, paragraf)
8. Cała odpowiedź musi być w języku POLSKIM
"""
        print(f"[RAG PIPELINE] STEP 3: Prompt built, length: {len(augmented_prompt)} chars")
        
        # Step 4: Generate
        print(f"[RAG PIPELINE] STEP 4: Invoking LLM...")
        try:
            print(f"[RAG] Invoking LLM with {len(augmented_prompt)} chars prompt...")
            response = llm_invoke_func(augmented_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            print(f"[RAG PIPELINE] STEP 4: LLM responded, answer length: {len(answer)} chars")
        except Exception as e:
            print(f"[RAG PIPELINE] STEP 4: ❌ LLM invocation failed: {str(e)}")
            answer = f"❌ Błąd LLM: {str(e)}\n\nZnalezione dane:\n{context}"
        
        # Step 5: Prepare metadata
        print(f"[RAG PIPELINE] STEP 5: Preparing metadata...")
        total_results = sum(len(results) for results in search_results.values())
        sources = list(search_results.keys())
        
        metadata = {
            "retrieval_success": True,
            "collections_searched": sources,
            "total_results": total_results,
            "results_per_collection": {coll: len(results) for coll, results in search_results.items()},
            "time_range_days": time_range_days
        }
        
        print(f"[RAG] ✓ Pipeline complete - {total_results} results from {len(sources)} collections")
        
        return {
            "answer": answer,
            "context": context,
            "sources": sources,
            "metadata": metadata,
            "raw_results": search_results  # For debugging
        }


def test_rag_engine():
    """Test RAG engine with sample queries."""
    from qdrant_client import QdrantClient
    
    # Setup
    client = QdrantClient(host="localhost", port=6333)
    ollama_url = "http://host.docker.internal:11434"
    embedding_model = "nomic-embed-text"
    
    rag = RAGEngine(client, ollama_url, embedding_model)
    
    # Test queries
    test_queries = [
        "Jakie są najczęstsze przyczyny eskalacji do człowieka?",
        "Ile mamy rozmów o stypendiach?",
        "Pokaż błędy techniczne z ostatnich 7 dni",
        "Jaki jest średni CSAT dla rozmów o egzaminach?"
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        # Test collection selection
        selected = rag.select_collections(query, top_k=3)
        print(f"Selected collections: {selected}")
        
        # Test search
        results = rag.multi_collection_search(query, collections=selected, limit_per_collection=5)
        
        # Format context
        context = rag.format_context_for_llm(results, max_results_per_collection=3)
        print(context)
        
        print(f"\nTotal results: {sum(len(r) for r in results.values())}")


if __name__ == "__main__":
    test_rag_engine()
