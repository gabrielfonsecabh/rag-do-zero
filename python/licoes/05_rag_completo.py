"""Lição 05: o RAG completo. Buscar, montar o prompt, gerar a resposta.

Sem chave de API, a lição para antes de chamar o modelo e mostra o prompt
exato que ele receberia. É o melhor jeito de depurar um RAG: quase todo erro
de resposta é um erro de contexto.

    python licoes/05_rag_completo.py "Quantos dias de férias posso vender?"

Com geração (escolha um):
    GERACAO_PROVEDOR=anthropic ANTHROPIC_API_KEY=... python licoes/05_rag_completo.py "..."
    GERACAO_PROVEDOR=openai OPENAI_API_KEY=... python licoes/05_rag_completo.py "..."
    GERACAO_PROVEDOR=openai OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \\
        GERACAO_MODELO=llama3.2 python licoes/05_rag_completo.py "..."   # Ollama, grátis
"""
import sys

import _base  # noqa: F401
from _base import titulo

from rag import criar_embedder, criar_gerador, indexar, responder

pergunta = " ".join(sys.argv[1:]) or "Quantos dias de férias posso vender?"
embedder = criar_embedder()
gerador = criar_gerador()
indice = indexar(embedder)

r = responder(pergunta, indice, embedder, gerador, k=4)

titulo("Trechos recuperados")
for i, (t, s) in enumerate(r.trechos, 1):
    print(f"[{i}] {s:.3f}  {t.arquivo} > {t.secao}")

if r.texto is None:
    titulo("Prompt que o modelo receberia (GERACAO_PROVEDOR=nenhum)")
    print(r.prompt)
else:
    titulo(f"Resposta ({gerador.nome})")
    print(r.texto)

print(
    "\nTeste uma pergunta que o manual NÃO responde, como 'Qual é o plano de saúde?'.\n"
    "Um bom RAG diz que não sabe. Se ele inventar, o problema está no prompt de\n"
    "sistema (rag/pipeline.py, SISTEMA) ou no modelo, não na busca."
)
