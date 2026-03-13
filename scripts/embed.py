"""
Gera embeddings dos resumos das normas e indexa no ChromaDB.
Lê do MongoDB, gera vetores com sentence-transformers, salva no Chroma.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB", "normas_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = "data/chroma_db"


def embed_normas():
    """Gera embeddings dos resumos e indexa no ChromaDB."""

    # Carrega modelo de embeddings
    print(f"Carregando modelo: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Conecta ao MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    normas = list(db["normas"].find({}))
    print(f"Encontradas {len(normas)} normas no MongoDB")

    # Inicializa ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Coleção de resumos (1 embedding por doc)
    resumos_col = chroma_client.get_or_create_collection(
        name="resumos",
        metadata={"hnsw:space": "cosine"}
    )

    # Gera embeddings dos resumos
    print("Gerando embeddings dos resumos...")
    ids, embeddings, metadatas, documents = [], [], [], []

    for norma in tqdm(normas):
        doc_id = str(norma["_id"])
        texto_resumo = f"{norma.get('titulo', '')}. {norma.get('resumo', '')}"

        emb = model.encode(texto_resumo).tolist()

        ids.append(doc_id)
        embeddings.append(emb)
        documents.append(texto_resumo)
        metadatas.append({
            "filename": norma.get("filename", ""),
            "titulo": norma.get("titulo", ""),
            "categoria": norma.get("categoria", "geral"),
            "tags": ", ".join(norma.get("tags", [])),
        })

    # Insere no ChromaDB
    resumos_col.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"✓ {len(ids)} embeddings indexados no ChromaDB ({CHROMA_DIR})")
    client.close()


if __name__ == "__main__":
    embed_normas()
