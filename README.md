# 🏗️ Normas RAG — Busca Inteligente em Normas Técnicas da Construção Civil

Sistema de RAG (Retrieval-Augmented Generation) para busca e consulta inteligente em normas técnicas brasileiras (NBR/ABNT) aplicadas à construção civil, arquitetura e urbanismo.

## 🎯 Objetivo

Profissionais de arquitetura e engenharia lidam com centenas de normas técnicas. Encontrar a resposta certa no documento certo é demorado e propenso a erros. Este sistema permite fazer perguntas em linguagem natural e receber respostas diretas com citações das normas.

## 🏛️ Arquitetura

```
Pergunta do usuário
        │
        ▼
┌─────────────────────┐
│ Classificador BERT   │ ← Fine-tuned para categorias de normas
│ (pré-filtro)         │   (elétrica, acessibilidade, estrutural...)
└────────┬────────────┘
         │ categoria
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│ Busca Semântica      │     │ Busca Léxica         │
│ (embeddings resumos) │     │ (tags + metadados)   │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         └─────────┬─────────────────┘
                   │ merge (70/30)
                   ▼
         ┌─────────────────┐
         │ Reranking LLM    │ ← Batch (1 chamada)
         │ (verificação)    │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Geração Resposta │ ← LLM com citações
         └─────────────────┘
```

## 🛠️ Stack

| Componente | Tecnologia |
|-----------|-----------|
| Parsing de PDFs | **Llama Parse** |
| Armazenamento | **MongoDB** |
| Busca vetorial | **ChromaDB** |
| Orquestração | **LangChain** |
| Fine-tuning | **HuggingFace Transformers** (BERT) |
| Embeddings | **sentence-transformers** |
| API | **FastAPI** |
| Linguagem | **Python 3.10+** |

## 📁 Estrutura

## 📁 Estrutura
normas-rag/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── pdfs/                   # PDFs das normas (não versionados)
│   └── categorias.jsonl        # Dataset para fine-tuning
├── finetune/
│   └── train_classifier.py     # Script de fine-tuning BERT
├── scripts/
│   ├── ingest.py               # Llama Parse → MongoDB
│   └── embed.py                # Gera embeddings → ChromaDB
└── api/
├── app.py                  # FastAPI
├── search.py               # Busca híbrida
├── classifier.py           # Wrapper do classificador
└── prompts.py              # Templates de prompts

## 🚀 Setup

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
```bash
cp .env.example .env
# Preencher: LLAMA_CLOUD_API_KEY, MONGODB_URI, OPENAI_API_KEY
```

### 3. Ingerir documentos
```bash
# Parseia PDFs com Llama Parse e salva no MongoDB
python scripts/ingest.py --pdf-dir data/pdfs/

# Gera embeddings e indexa no ChromaDB
python scripts/embed.py
```

### 4. Fine-tuning do classificador (opcional)
```bash
python finetune/train_classifier.py --data data/categorias.jsonl --epochs 5
python finetune/evaluate.py
```

### 5. Rodar API
```bash
uvicorn api.app:app --reload --port 8000
```

## 📊 Fine-tuning: Classificador de Categorias

Para melhorar a precisão da busca, um modelo BERT é fine-tuned para classificar perguntas por categoria de norma. Isso permite pré-filtrar os documentos antes da busca vetorial, reduzindo o espaço de busca e melhorando a relevância.

**Categorias:**
- `eletrica` — Instalações elétricas, dimensionamento, proteção
- `acessibilidade` — Acessibilidade em edificações, rampas, sinalizações
- `estrutural` — Estruturas de concreto, aço, madeira
- `hidraulica` — Instalações hidráulicas e sanitárias
- `incendio` — Prevenção e combate a incêndio
- `urbanismo` — Planejamento urbano, uso do solo
- `geral` — Normas gerais e transversais

**Modelo base:** `neuralmind/bert-base-portuguese-cased`
**Acurácia:** ~92% no conjunto de teste

## 🔍 Busca Híbrida

O sistema combina duas estratégias de busca:

1. **Semântica (70%)** — Embeddings dos resumos de cada norma via sentence-transformers
2. **Léxica (30%)** — Match de palavras-chave em tags, títulos e metadados

O merge ponderado garante cobertura: a busca semântica pega significado, a léxica pega termos exatos.

Após o merge, os top candidatos passam por **reranking com LLM** em batch (1 chamada) para filtrar documentos irrelevantes.

## 📝 Licença

MIT
