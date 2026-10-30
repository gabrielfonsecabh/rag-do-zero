"""Armazenamento em Postgres com pgvector (lição 07).

Usa psycopg e manda o vetor como texto ('[0.1,0.2,...]'::vector), sem
dependência extra. O esquema está em sql/0001_rag.sql.
"""

from __future__ import annotations

from pathlib import Path

from .chunking import Trecho
from .indice import IndiceIncompativel
from .provedores import Embedder

ARQUIVO_SQL = Path(__file__).resolve().parents[2] / "sql" / "0001_rag.sql"


def _vetor_sql(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def conectar(url: str):
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError("Instale o driver: pip install 'psycopg[binary]'") from e
    return psycopg.connect(url, autocommit=True)


def criar_esquema(conn) -> None:
    conn.execute(ARQUIVO_SQL.read_text(encoding="utf-8"))


def _dimensao_da_tabela(conn) -> int:
    linha = conn.execute(
        """select atttypmod from pg_attribute
            where attrelid = 'rag.trechos'::regclass and attname = 'embedding'"""
    ).fetchone()
    return int(linha[0])


def indexar(conn, colecao: str, trechos: list[Trecho], embedder: Embedder) -> int:
    vetores = embedder.embed_documentos([t.texto_para_embedding for t in trechos])
    dim = len(vetores[0]) if vetores else 0
    if vetores and dim != _dimensao_da_tabela(conn):
        raise IndiceIncompativel(
            f"O modelo gera {dim} dimensões e a tabela foi criada com "
            f"{_dimensao_da_tabela(conn)}. Ajuste vector(N) em sql/0001_rag.sql e recrie."
        )
    with conn.transaction():
        conn.execute("delete from rag.colecoes where id = %s", (colecao,))
        conn.execute(
            "insert into rag.colecoes (id, embedder, dimensao) values (%s, %s, %s)",
            (colecao, embedder.nome, dim),
        )
        with conn.cursor() as cur:
            cur.executemany(
                """insert into rag.trechos (colecao, arquivo, secao, texto, embedding)
                   values (%s, %s, %s, %s, %s::vector)""",
                [(colecao, t.arquivo, t.secao, t.texto, _vetor_sql(v)) for t, v in zip(trechos, vetores)],
            )
    return len(trechos)


def conferir(conn, colecao: str, embedder: Embedder) -> None:
    linha = conn.execute("select embedder from rag.colecoes where id = %s", (colecao,)).fetchone()
    if linha is None:
        raise IndiceIncompativel(f"A coleção '{colecao}' não existe. Indexe primeiro.")
    if linha[0] != embedder.nome:
        raise IndiceIncompativel(
            f"A coleção '{colecao}' foi indexada com '{linha[0]}' e a busca está usando "
            f"'{embedder.nome}'. Reindexe com o modelo novo."
        )


def buscar(conn, colecao: str, pergunta: str, embedder: Embedder, k: int = 4, hibrida: bool = True):
    conferir(conn, colecao, embedder)
    v = _vetor_sql(embedder.embed_pergunta(pergunta))
    if hibrida:
        sql = "select arquivo, secao, texto, score from rag.buscar_hibrido(%s, %s, %s::vector, %s)"
        params = (colecao, pergunta, v, k)
    else:
        sql = "select arquivo, secao, texto, similaridade from rag.buscar(%s, %s::vector, %s)"
        params = (colecao, v, k)
    return [
        (Trecho(id=f"{a}#pg", arquivo=a, secao=s, texto=t), float(score))
        for a, s, t, score in conn.execute(sql, params).fetchall()
    ]
