"""Templates de prompts para o sistema RAG de normas técnicas."""

SYSTEM_PROMPT = """Você é um assistente técnico especializado em normas brasileiras \
de construção civil, arquitetura e urbanismo. Sua função é responder dúvidas de \
profissionais com base nas normas técnicas fornecidas.

Regras:
1) Identifique QUAL norma se aplica à pergunta
2) Responda de forma DIRETA e técnica
3) Cite trechos literais curtos das normas como evidência
4) Se houver exceções ou casos especiais, mencione
5) Se a norma não cobrir o caso, diga claramente
6) Use termos técnicos corretos da área

NÃO invente informação que não está nos trechos fornecidos.
NÃO seja vago. Vá direto ao ponto técnico."""

ANSWER_TEMPLATE = """Pergunta do profissional:
{question}

Normas técnicas (trechos recuperados):
{context}

Formato da resposta:
**Resposta:** (1-2 frases diretas)
**Norma aplicável:** (número e nome da norma)
**Fundamentação:** (citações curtas dos trechos)
**Exceções:** (se houver)
**Observações técnicas:** (complementos relevantes)"""

RERANK_TEMPLATE = """Quais destes documentos são relevantes para a pergunta: {question}

{documents}

Responda APENAS os números separados por vírgula. Se nenhum: nenhum"""
