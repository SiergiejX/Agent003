#!/usr/bin/env python3
"""
Skrypt do importu kolekcji Qdrant z plików JSON.
Odtwarza dane wyeksportowane przez export_qdrant_collections.py
"""

import json
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Połączenie z Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Katalog z eksportami
EXPORT_DIR = Path("./qdrant_export")


def import_collection(json_file: Path, overwrite: bool = False):
    """Importuje kolekcję z pliku JSON."""
    print(f"\n{'='*60}")
    print(f"Importowanie z pliku: {json_file.name}")
    print(f"{'='*60}")
    
    try:
        # Wczytaj dane z pliku
        print("Wczytywanie danych z pliku...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        collection_name = data["collection_name"]
        points_data = data["points"]
        collection_info = data["collection_info"]
        
        print(f"Kolekcja: {collection_name}")
        print(f"Punktów do importu: {len(points_data)}")
        print(f"Data eksportu: {data['export_timestamp']}")
        
        # Sprawdź czy kolekcja już istnieje
        try:
            existing = client.get_collection(collection_name)
            if overwrite:
                print(f"⚠ Usuwam istniejącą kolekcję '{collection_name}'...")
                client.delete_collection(collection_name)
            else:
                print(f"✗ Kolekcja '{collection_name}' już istnieje!")
                print("  Użyj parametru --overwrite aby nadpisać")
                return {
                    "collection": collection_name,
                    "success": False,
                    "error": "Collection already exists"
                }
        except:
            pass  # Kolekcja nie istnieje, można kontynuować
        
        # Utwórz kolekcję
        print(f"Tworzę kolekcję '{collection_name}'...")
        vector_size = 768
        
        # Zamień string distance na enum
        distance_str = collection_info["config"]["params"]["vectors"]["distance"]
        if "COSINE" in distance_str:
            distance = Distance.COSINE
        elif "EUCLID" in distance_str:
            distance = Distance.EUCLID
        elif "DOT" in distance_str:
            distance = Distance.DOT
        else:
            distance = Distance.COSINE  # domyślnie
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        
        print(f"✓ Kolekcja utworzona (vector_size={vector_size}, distance={distance})")
        
        # Importuj punkty w partiach
        batch_size = 100
        total_imported = 0
        
        for i in range(0, len(points_data), batch_size):
            batch = points_data[i:i + batch_size]
            
            # Konwertuj z dict na PointStruct
            points = []
            for point_data in batch:
                # Konwertuj id - może być stringiem lub liczbą
                point_id = point_data["id"]
                try:
                    point_id = int(point_id)
                except (ValueError, TypeError):
                    # Jeśli nie można skonwertować na int, użyj hasha
                    point_id = hash(str(point_id)) % (10 ** 8)
                
                point = PointStruct(
                    id=point_id,
                    vector=point_data["vector"],
                    payload=point_data["payload"]
                )
                points.append(point)
            
            # Wstaw partię
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            total_imported += len(points)
            print(f"  Zaimportowano {total_imported}/{len(points_data)} punktów...", end="\r")
        
        print(f"\n✓ Zaimportowano {total_imported} punktów")
        
        # Weryfikuj
        collection = client.get_collection(collection_name)
        print(f"✓ Weryfikacja: kolekcja zawiera {collection.points_count} punktów")
        
        return {
            "collection": collection_name,
            "points": total_imported,
            "success": True
        }
        
    except Exception as e:
        print(f"✗ Błąd podczas importu: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "collection": json_file.stem,
            "success": False,
            "error": str(e)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Import Qdrant collections from JSON files'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing collections'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Import specific file (otherwise imports all)'
    )
    
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║  IMPORT KOLEKCJI QDRANT                                    ║
║  Importuje dane z plików JSON do Qdrant                    ║
╚════════════════════════════════════════════════════════════╝
""")
    
    print(f"Katalog importu: {EXPORT_DIR.absolute()}")
    print(f"Połączenie: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Nadpisywanie: {'TAK' if args.overwrite else 'NIE'}")
    
    # Sprawdź czy katalog istnieje
    if not EXPORT_DIR.exists():
        print(f"\n✗ Katalog {EXPORT_DIR} nie istnieje!")
        print("Najpierw wyeksportuj dane używając export_qdrant_collections.py")
        return
    
    # Znajdź pliki JSON do importu
    if args.file:
        json_files = [Path(args.file)]
    else:
        json_files = sorted(EXPORT_DIR.glob("agent*.json"))
        json_files += sorted(EXPORT_DIR.glob("Baza*.json"))
    
    if not json_files:
        print(f"\n✗ Nie znaleziono plików JSON w {EXPORT_DIR}")
        return
    
    print(f"\nZnaleziono {len(json_files)} plików do importu:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Importuj każdy plik
    results = []
    for json_file in json_files:
        result = import_collection(json_file, overwrite=args.overwrite)
        results.append(result)
    
    # Podsumowanie
    print(f"\n{'='*60}")
    print("PODSUMOWANIE IMPORTU")
    print(f"{'='*60}")
    
    total_points = 0
    successful = 0
    
    for result in results:
        if result["success"]:
            successful += 1
            total_points += result.get("points", 0)
            print(f"✓ {result['collection']}: {result.get('points', 0)} punktów")
        else:
            print(f"✗ {result['collection']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*60}")
    print(f"Zaimportowano: {successful}/{len(results)} kolekcji")
    print(f"Łącznie punktów: {total_points}")
    print(f"{'='*60}")
    
    if successful == len(results):
        print("\n✓ Import zakończony pomyślnie!")
    else:
        print("\n⚠ Niektóre kolekcje nie zostały zaimportowane")


if __name__ == "__main__":
    main()
