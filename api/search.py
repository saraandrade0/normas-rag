"""
Busca híbrida: semântica (ChromaDB) + léxica (MongoDB).
Merge ponderado + reranking com LLM.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_openai import ChatOpenAI

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB", "normas_rag")
CHROMA_DIR = "data/chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class HybridSearch:
    """
    Busca híbrida combinando:
    - Busca semântica por resumos (ChromaDB)
    - Busca léxica por tags/títulos (MongoDB)
    - Merge ponderado (70% semântico, 30% léxico)
    - Reranking com LLM em batch
    """

    def __init__(self):
        # MongoDB
        self.mongo = MongoClient(MONGO_URI)
        self.db = self.mongo[MONGO_DB]
        self.normas = self.db["normas"]

        # ChromaDB
        self.chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        self.resumos_col = self.chroma.get_collection("resumos")

        # Embedding model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # LLM para reranking
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def busca_semantica(self, query: str, categoria: str = None, top_n: int = 100) -> dict:
        """Busca semântica nos resumos via ChromaDB."""
        query_emb = self.embedder.encode(query).tolist()

        where_filter = {"categoria": categoria} if categoria else None

        results = self.resumos_col.query(
            query_embeddings=[query_emb],
            n_results=top_n,
            where=where_filter,
        )

        scores = {}
        if results["ids"] and results["distances"]:
            for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB retorna distância; converte pra similaridade
                scores[doc_id] = 1 - dist
        return scores

    def busca_lexica(self, query: str, categoria: str = None) -> dict:
        """Busca léxica por tags e títulos no MongoDB."""
        termos = query.lower().split()

        filtro = {}
        if categoria:
            filtro["categoria"] = categoria

        normas = list(self.normas.find(filtro))
        scores = {}

        for norma in normas:
            doc_id = str(norma["_id"])
            titulo = norma.get("titulo", "").lower()
            resumo = norma.get("resumo", "").lower()
            tags = " ".join(norma.get("tags", [])).lower()
            tudo = f"{titulo} {resumo} {tags}"

            matches = sum(1 for t in termos if t in tudo and len(t) > 3)
            if matches > 0:
                scores[doc_id] = matches / len(termos)

        return scores

    def merge_rankings(self, scores_sem: dict, scores_lex: dict, top_n: int = 30) -> list:
        """Merge ponderado: 70% semântico + 30% léxico."""
        todos = set(scores_sem.keys()) | set(scores_lex.keys())
        ranking = []

        for doc_id in todos:
            s_sem = scores_sem.get(doc_id, 0)
            s_lex = scores_lex.get(doc_id, 0)
            score = 0.7 * s_sem + 0.3 * s_lex
            ranking.append((doc_id, score))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranking[:top_n]]

    def rerank_batch(self, query: str, doc_ids: list) -> list:
        """Reranking com LLM em batch — 1 chamada para todos os candidatos."""
        from bson import ObjectId

        docs_info = []
        for i, doc_id in enumerate(doc_ids):
            norma = self.normas.find_one({"_id": ObjectId(doc_id)})
            if norma:
                docs_info.append(
                    f"[{i}] {norma.get('titulo', '')} | "
                    f"Tags: {', '.join(norma.get('tags', []))} | "
                    f"Resumo: {norma.get('resumo', '')[:200]}"
                )

        bloco = "\n".join(docs_info)
        prompt = (
            f"Quais destes documentos são relevantes para a pergunta: {query}\n\n"
            f"{bloco}\n\n"
            f"Responda APENAS os números separados por vírgula. Se nenhum: nenhum"
        )

        resp = self.llm.invoke(prompt)
        content = resp.content.strip()

        if "nenhum" in content.lower():
            return []

        nums = []
        for p in content.replace(",", " ").split():
            try:
                n = int(p)
                if 0 <= n < len(doc_ids):
                    nums.append(doc_ids[n])
            except ValueError:
                pass
        return nums

    def search(self, query: str, categoria: str = None, max_docs: int = 10) -> list:
        """
        Pipeline completo de busca:
        1. Busca semântica (resumos)
        2. Busca léxica (tags)
        3. Merge ponderado (70/30)
        4. Reranking LLM
        5. Retorna documentos completos
        """
        from bson import ObjectId

        # Etapa 1 + 2: Buscas paralelas
        scores_sem = self.busca_semantica(query, categoria, top_n=100)
        scores_lex = self.busca_lexica(query, categoria)

        # Etapa 3: Merge
        top_ids = self.merge_rankings(scores_sem, scores_lex, top_n=30)

        if not top_ids:
            return []

        # Etapa 4: Rerank
        filtrados = self.rerank_batch(query, top_ids)
        if not filtrados:
            filtrados = top_ids[:max_docs]

        # Etapa 5: Busca docs completos no MongoDB
        resultados = []
        for doc_id in filtrados[:max_docs]:
            norma = self.normas.find_one({"_id": ObjectId(doc_id)})
            if norma:
                norma["_id"] = str(norma["_id"])
                resultados.append(norma)

        return resultados
