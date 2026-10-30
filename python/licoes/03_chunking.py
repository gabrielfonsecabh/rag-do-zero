"""Lição 03: dividir documentos em trechos (chunking).

O modelo de embedding resume um texto inteiro num único vetor. Texto grande
demais mistura assuntos e o vetor vira uma média sem foco; texto pequeno
demais perde o contexto. Chunking é escolher esse tamanho.

    python licoes/03_chunking.py
    python licoes/03_chunking.py --max 300 --sobreposicao 60
"""
import argparse

import _base  # noqa: F401
from _base import titulo

from rag import dividir_markdown
from rag.pipeline import carregar_documentos

p = argparse.ArgumentParser()
p.add_argument("--max", type=int, default=800, help="tamanho máximo do trecho, em caracteres")
p.add_argument("--sobreposicao", type=int, default=150)
args = p.parse_args()

trechos = []
for nome, texto in carregar_documentos():
    trechos += dividir_markdown(texto, nome, args.max, args.sobreposicao)

titulo(f"{len(trechos)} trechos com máximo de {args.max} caracteres")
tamanhos = sorted(len(t.texto) for t in trechos)
print(f"menor {tamanhos[0]}, mediana {tamanhos[len(tamanhos) // 2]}, maior {tamanhos[-1]}")

titulo("Três trechos de exemplo")
for t in trechos[:3]:
    print(f"[{t.id}] seção: {t.secao}\n{t.texto}\n")

print(
    "Cada trecho guarda o caminho da seção ('Férias > Abono'). Ele vai junto para\n"
    "o embedding, porque o título é a melhor pista de assunto que o texto tem.\n"
    "Teste --max 200 e veja trechos quebrados no meio de uma ideia; a\n"
    "sobreposição repete o fim de um trecho no começo do próximo para compensar."
)
