"""
Ingestão de PDFs de normas técnicas.
Llama Parse → extrai texto estruturado dos PDFs
MongoDB → armazena documentos, resumos e metadados
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from llama_parse import LlamaParse
from pymongo import MongoClient
from langchain_openai import ChatOpenAI

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB", "normas_rag")
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


def parse_pdf(pdf_path: str) -> dict:
    """Parseia um PDF com Llama Parse e retorna texto estruturado."""
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="markdown",
        language="pt",
        parsing_instruction=(
            "Este é um documento de norma técnica brasileira (NBR/ABNT). "
            "Extraia o texto completo preservando a estrutura de seções, "
            "tabelas e listas numeradas."
        )
    )
    documents = parser.load_data(pdf_path)
    full_text = "\n\n".join([doc.text for doc in documents])
    return {
        "filename": os.path.basename(pdf_path),
        "text": full_text,
        "pages": len(documents),
    }


def generate_summary_and_tags(text: str, filename: str) -> dict:
    """Usa LLM para gerar resumo e tags de uma norma."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Analise esta norma técnica e retorne um JSON com:
- "titulo": título oficial da norma
- "resumo": resumo de 2-3 frases do conteúdo
- "categoria": uma de [eletrica, acessibilidade, estrutural, hidraulica, incendio, urbanismo, geral]
- "tags": lista de 5-10 palavras-chave relevantes

Arquivo: {filename}
Texto (primeiros 3000 chars):
{text[:3000]}

Responda APENAS o JSON, sem markdown."""

    resp = llm.invoke(prompt)
    try:
        return json.loads(resp.content)
    except json.JSONDecodeError:
        return {
            "titulo": filename,
            "resumo": text[:200],
            "categoria": "geral",
            "tags": []
        }


def ingest_pdfs(pdf_dir: str):
    """Pipeline completo: PDF → Llama Parse → LLM (resumo/tags) → MongoDB."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db["normas"]

    # Limpa coleção existente
    collection.drop()
    collection.create_index("filename", unique=True)

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    print(f"Encontrados {len(pdf_files)} PDFs em {pdf_dir}")

    for pdf_path in tqdm(pdf_files, desc="Ingerindo normas"):
        try:
            # 1. Parse com Llama Parse
            parsed = parse_pdf(str(pdf_path))

            # 2. Gera resumo e tags com LLM
            metadata = generate_summary_and_tags(parsed["text"], parsed["filename"])

            # 3. Salva no MongoDB
            doc = {
                "filename": parsed["filename"],
                "text": parsed["text"],
                "pages": parsed["pages"],
                "titulo": metadata.get("titulo", ""),
                "resumo": metadata.get("resumo", ""),
                "categoria": metadata.get("categoria", "geral"),
                "tags": metadata.get("tags", []),
            }
            collection.insert_one(doc)
            print(f"  ✓ {parsed['filename']} → {metadata.get('categoria', '?')}")

        except Exception as e:
            print(f"  ✗ {pdf_path.name}: {e}")

    count = collection.count_documents({})
    print(f"\nTotal: {count} normas ingeridas no MongoDB")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão de normas técnicas")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="Diretório com PDFs")
    args = parser.parse_args()
    ingest_pdfs(args.pdf_dir)
