"""Lição 07: levar o RAG para produção com Postgres + pgvector.

O índice em JSON das lições anteriores cabe na memória e serve para aprender.
Em produção, com milhares de documentos e várias pessoas consultando, o lugar
dos vetores é o banco. Funciona com Supabase (que já traz pgvector) ou com
qualquer Postgres 15+ com a extensão instalada.

    pip install 'psycopg[binary]'
    export DATABASE_URL=postgresql://usuario:senha@host:5432/banco
    python licoes/07_pgvector.py indexar
    python licoes/07_pgvector.py buscar "Qual o limite de hospedagem?"

Antes, ajuste vector(384) em sql/0001_rag.sql se o seu modelo tiver outra
dimensão. A busca usa a função rag.buscar_hibrido, que junta semântica e
palavra-chave.
"""
import os
import sys

import _base  # noqa: F401
from _base import titulo

from rag import criar_embedder, dividir_markdown
from rag import postgres as pg
from rag.pipeline import carregar_documentos

url = os.environ.get("DATABASE_URL")
if not url:
    sys.exit("Defina DATABASE_URL com a conexão do Postgres (no Supabase: Project Settings > Database).")

acao = sys.argv[1] if len(sys.argv) > 1 else "buscar"
embedder = criar_embedder()
conn = pg.conectar(url)

if acao == "indexar":
    pg.criar_esquema(conn)
    trechos = []
    for nome, texto in carregar_documentos():
        trechos += dividir_markdown(texto, nome)
    n = pg.indexar(conn, "manual", trechos, embedder)
    print(f"{n} trechos gravados na coleção 'manual' com {embedder.nome}")
else:
    pergunta = " ".join(sys.argv[2:]) or "Qual o limite de hospedagem?"
    titulo(f"Busca híbrida: {pergunta}")
    for t, score in pg.buscar(conn, "manual", pergunta, embedder, k=4):
        print(f"{score:.4f}  {t.arquivo} > {t.secao}")
