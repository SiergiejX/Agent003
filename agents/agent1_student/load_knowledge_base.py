#!/usr/bin/env python3
"""
Load knowledge base documents into Qdrant.
Processes .txt, .pdf, .docx, .doc files from chatbot-baza-wiedzy-nowa/
"""

import os
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Document processing
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️ python-docx not installed - .docx files will be skipped")

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("⚠️ PyPDF2 not installed - .pdf files will be skipped")


# Qdrant connection
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "BazaWiedzy"
EMBEDDING_DIM = 768

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def generate_simple_embedding(text: str, dim: int = 768) -> List[float]:
    """Generate simple hash-based embedding."""
    hash_obj = hashlib.sha256(text.lower().encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 255.0) * 2 - 1)
    
    return embedding


def read_txt_file(filepath: Path) -> str:
    """Read text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try different encoding
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()


def read_docx_file(filepath: Path) -> str:
    """Read .docx file."""
    if not HAS_DOCX:
        return f"[Dokument DOCX: {filepath.name} - wymagana biblioteka python-docx]"
    
    try:
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        return f"[Błąd odczytu DOCX: {e}]"


def read_pdf_file(filepath: Path) -> str:
    """Read .pdf file."""
    if not HAS_PDF:
        return f"[Dokument PDF: {filepath.name} - wymagana biblioteka PyPDF2]"
    
    try:
        text = []
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    except Exception as e:
        return f"[Błąd odczytu PDF: {e}]"


def read_doc_file(filepath: Path) -> str:
    """Read .doc file (old Word format)."""
    # For .doc files, we'll use a simple placeholder
    # In production, you might want to use antiword or textract
    return f"[Dokument DOC: {filepath.name} - wymaga konwersji do DOCX lub instalacji antiword]"


def read_document(filepath: Path) -> str:
    """Read document based on extension."""
    ext = filepath.suffix.lower()
    
    if ext == '.txt':
        return read_txt_file(filepath)
    elif ext == '.docx':
        return read_docx_file(filepath)
    elif ext == '.pdf':
        return read_pdf_file(filepath)
    elif ext == '.doc':
        return read_doc_file(filepath)
    else:
        return f"[Nieobsługiwany format: {ext}]"


def create_collection():
    """Create Qdrant collection for knowledge base."""
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"✓ Kolekcja '{COLLECTION_NAME}' już istnieje")
        
        # Delete and recreate
        response = input("Czy chcesz usunąć i przetworzyć ponownie? (y/n): ")
        if response.lower() == 'y':
            client.delete_collection(COLLECTION_NAME)
            print(f"✓ Usunięto kolekcję '{COLLECTION_NAME}'")
        else:
            print("Anulowano")
            return False
    except:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    print(f"✓ Utworzono kolekcję '{COLLECTION_NAME}'")
    return True


def load_documents(base_dir: Path) -> List[Dict[str, Any]]:
    """Load all documents from knowledge base directory."""
    documents = []
    categories = ["dane_osobowe", "egzaminy", "rekrutacja", "stypendia", "urlopy_zwolnienia"]
    
    for category in categories:
        category_path = base_dir / category
        if not category_path.exists():
            print(f"⚠️ Brak katalogu: {category}")
            continue
        
        print(f"\n📁 Kategoria: {category}")
        
        for filepath in category_path.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in ['.txt', '.docx', '.pdf', '.doc']:
                print(f"  📄 {filepath.name}...", end=" ")
                
                content = read_document(filepath)
                
                if content and len(content.strip()) > 10:
                    documents.append({
                        "filename": filepath.name,
                        "category": category,
                        "content": content,
                        "filepath": str(filepath.relative_to(base_dir))
                    })
                    print(f"✓ ({len(content)} znaków)")
                else:
                    print(f"⚠️ Pusty lub zbyt krótki")
    
    return documents


def upload_to_qdrant(documents: List[Dict[str, Any]]):
    """Upload documents to Qdrant."""
    print(f"\n📤 Uploading {len(documents)} dokumentów do Qdrant...")
    
    points = []
    for idx, doc in enumerate(documents, 1):
        # Generate embedding from content
        embedding = generate_simple_embedding(doc["content"], dim=EMBEDDING_DIM)
        
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "filename": doc["filename"],
                "category": doc["category"],
                "content": doc["content"][:1000],  # First 1000 chars for payload
                "full_content": doc["content"],
                "filepath": doc["filepath"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "doc_type": "knowledge_base"
            }
        )
        points.append(point)
    
    # Upload in batch
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✓ Uploaded {len(points)} dokumentów")


def main():
    """Main function."""
    print("="*60)
    print("ŁADOWANIE BAZY WIEDZY DO QDRANT")
    print("="*60)
    
    # Base directory
    base_dir = Path(__file__).parent / "chatbot-baza-wiedzy-nowa"
    
    if not base_dir.exists():
        print(f"❌ Nie znaleziono katalogu: {base_dir}")
        return
    
    print(f"📂 Katalog bazowy: {base_dir}\n")
    
    # Create collection
    if not create_collection():
        return
    
    # Load documents
    documents = load_documents(base_dir)
    
    if not documents:
        print("\n❌ Nie znaleziono żadnych dokumentów")
        return
    
    print(f"\n✓ Załadowano {len(documents)} dokumentów")
    
    # Upload to Qdrant
    upload_to_qdrant(documents)
    
    # Summary
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    
    info = client.get_collection(COLLECTION_NAME)
    print(f"✓ Kolekcja: {COLLECTION_NAME}")
    print(f"✓ Liczba dokumentów: {info.points_count}")
    print(f"✓ Status: {info.status}")
    
    # Category breakdown
    categories = {}
    for doc in documents:
        cat = doc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📊 Dokumenty wg kategorii:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count}")
    
    print("\n✅ Baza wiedzy gotowa!")


if __name__ == "__main__":
    main()
