"""
API — Busca Inteligente em Normas Técnicas
FastAPI + Busca Híbrida + Classificador BERT + LLM
"""

import time
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from api.search import HybridSearch
from api.classifier import NormaClassifier
from api.prompts import SYSTEM_PROMPT, ANSWER_TEMPLATE
load_dotenv()

# ─── INIT ────────────────────────────────────────────────
app = FastAPI(title="Normas RAG", version="1.0")

print("Inicializando...")
search_engine = HybridSearch()
classifier = NormaClassifier()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("✓ Pronto")


# ─── SCHEMAS ─────────────────────────────────────────────
class Pergunta(BaseModel):
    question: str

class Fonte(BaseModel):
    titulo: str
    filename: str
    resumo: str
    categoria: str
    trecho: str

class Resposta(BaseModel):
    answer: str
    categoria_detectada: str | None
    fontes: list[Fonte]
    tempo: float
    n_docs: int


# ─── ENDPOINT ────────────────────────────────────────────
@app.post("/api/busca", response_model=Resposta)
async def busca(req: Pergunta):
    t0 = time.time()
    question = req.question.strip()

    # 1. Classificação (pré-filtro)
    categoria = classifier.classify(question)

    # 2. Busca híbrida (com filtro de categoria se disponível)
    docs = search_engine.search(question, categoria=categoria, max_docs=10)

    # 3. Monta contexto
    fontes = []
    context_blocks = []
    for doc in docs:
        fontes.append(Fonte(
            titulo=doc.get("titulo", ""),
            filename=doc.get("filename", ""),
            resumo=doc.get("resumo", ""),
            categoria=doc.get("categoria", ""),
            trecho=doc.get("text", "")[:1500],
        ))
        context_blocks.append(
            f"### {doc.get('titulo', '')}\n"
            f"Arquivo: {doc.get('filename', '')}\n"
            f"Resumo: {doc.get('resumo', '')}\n"
            f"Texto:\n{doc.get('text', '')[:3000]}\n"
        )

    context = "\n\n---\n\n".join(context_blocks)

    # 4. Gera resposta com LLM
    prompt = ANSWER_TEMPLATE.format(question=question, context=context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    resp = llm.invoke(messages)

    return Resposta(
        answer=resp.content,
        categoria_detectada=categoria,
        fontes=fontes,
        tempo=round(time.time() - t0, 2),
        n_docs=len(fontes),
    )


# ─── FRONTEND ────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")
