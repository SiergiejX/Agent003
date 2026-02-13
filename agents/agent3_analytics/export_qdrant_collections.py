#!/usr/bin/env python3
"""
Skrypt do eksportu wszystkich kolekcji z Qdrant do plików JSON.
Umożliwia przeniesienie danych na inną maszynę.
"""

import json
import os
from datetime import datetime
from qdrant_client import QdrantClient
from pathlib import Path

# Połączenie z Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Kolekcje do eksportu
COLLECTIONS = [
    "agent1_conversations",
    "agent1_turns",
    "BazaWiedzy",
    "agent2_tickets",
    "agent3_analitics",
    "agent4_bos_querries"
]

# Katalog na eksport
EXPORT_DIR = Path("./qdrant_export")
EXPORT_DIR.mkdir(exist_ok=True)

def export_collection(collection_name: str):
    """Eksportuje pojedynczą kolekcję do pliku JSON."""
    print(f"\n{'='*60}")
    print(f"Eksportowanie kolekcji: {collection_name}")
    print(f"{'='*60}")
    
    try:
        # Pobierz informacje o kolekcji
        collection_info = client.get_collection(collection_name)
        print(f"Liczba punktów: {collection_info.points_count}")
        print(f"Status: {collection_info.status}")
        
        # Pobierz wszystkie punkty (w partiach)
        all_points = []
        offset = None
        batch_size = 100
        batch_num = 0
        
        while True:
            batch_num += 1
            print(f"  Pobieranie partii {batch_num}... ", end="", flush=True)
            
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                print("brak danych")
                break
            
            print(f"{len(points)} punktów")
            
            # Konwertuj punkty do słowników
            for point in points:
                point_dict = {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": point.payload
                }
                all_points.append(point_dict)
            
            # Jeśli nie ma więcej danych, zakończ
            if next_offset is None:
                break
            
            offset = next_offset
        
        # Zapisz do pliku JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = EXPORT_DIR / f"{collection_name}_{timestamp}.json"
        
        export_data = {
            "collection_name": collection_name,
            "export_timestamp": datetime.now().isoformat(),
            "total_points": len(all_points),
            "collection_info": {
                "status": collection_info.status,
                "vectors_count": collection_info.points_count,
                "config": {
                    "params": {
                        "vectors": {
                            "size": collection_info.config.params.vectors.size,
                            "distance": str(collection_info.config.params.vectors.distance)
                        }
                    }
                }
            },
            "points": all_points
        }
        
        print(f"\nZapisywanie do pliku: {filename}")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        # Sprawdź rozmiar pliku
        file_size = os.path.getsize(filename)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✓ Zapisano {len(all_points)} punktów")
        print(f"✓ Rozmiar pliku: {file_size_mb:.2f} MB")
        
        return {
            "collection": collection_name,
            "points": len(all_points),
            "filename": str(filename),
            "size_mb": file_size_mb,
            "success": True
        }
        
    except Exception as e:
        print(f"✗ Błąd podczas eksportu: {str(e)}")
        return {
            "collection": collection_name,
            "success": False,
            "error": str(e)
        }


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║  EKSPORT KOLEKCJI QDRANT                                   ║
║  Eksportuje wszystkie dane z kolekcji do plików JSON       ║
╚════════════════════════════════════════════════════════════╝
""")
    
    print(f"Katalog eksportu: {EXPORT_DIR.absolute()}")
    print(f"Połączenie: {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Lista dostępnych kolekcji
    try:
        collections = client.get_collections()
        available_collections = [c.name for c in collections.collections]
        print(f"\nDostępne kolekcje: {', '.join(available_collections)}")
    except Exception as e:
        print(f"✗ Błąd połączenia z Qdrant: {e}")
        return
    
    # Eksportuj każdą kolekcję
    results = []
    for collection_name in COLLECTIONS:
        if collection_name in available_collections:
            result = export_collection(collection_name)
            results.append(result)
        else:
            print(f"\n⚠ Kolekcja '{collection_name}' nie istnieje, pomijam")
            results.append({
                "collection": collection_name,
                "success": False,
                "error": "Collection not found"
            })
    
    # Podsumowanie
    print(f"\n{'='*60}")
    print("PODSUMOWANIE EKSPORTU")
    print(f"{'='*60}")
    
    total_points = 0
    total_size = 0
    successful = 0
    
    for result in results:
        if result["success"]:
            successful += 1
            total_points += result["points"]
            total_size += result["size_mb"]
            print(f"✓ {result['collection']}: {result['points']} punktów, {result['size_mb']:.2f} MB")
        else:
            print(f"✗ {result['collection']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*60}")
    print(f"Wyeksportowano: {successful}/{len(results)} kolekcji")
    print(f"Łącznie punktów: {total_points}")
    print(f"Łączny rozmiar: {total_size:.2f} MB")
    print(f"{'='*60}")
    
    # Zapisz manifest eksportu
    manifest_file = EXPORT_DIR / f"export_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest = {
        "export_date": datetime.now().isoformat(),
        "source_host": f"{QDRANT_HOST}:{QDRANT_PORT}",
        "collections": results,
        "total_points": total_points,
        "total_size_mb": total_size
    }
    
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Zapisano manifest: {manifest_file}")
    print(f"\n📦 Pliki gotowe do przeniesienia znajdują się w: {EXPORT_DIR.absolute()}")
    print("\nAby zaimportować dane na innej maszynie, użyj skryptu import_qdrant_collections.py")


if __name__ == "__main__":
    main()
