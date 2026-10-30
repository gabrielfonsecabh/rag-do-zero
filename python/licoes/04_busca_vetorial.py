"""Lição 04: busca vetorial.

Indexar = calcular o vetor de cada trecho uma vez e guardar. Buscar = calcular
o vetor da pergunta e pegar os trechos mais próximos.

    python licoes/04_busca_vetorial.py "Posso trabalhar de outro país?"
    EMBEDDINGS_PROVEDOR=brinquedo python licoes/04_busca_vetorial.py "Perdi meu computador"
"""
import sys
import time
from pathlib import Path

import _base  # noqa: F401
from _base import titulo

from rag import Indice, criar_embedder, indexar

pergunta = " ".join(sys.argv[1:]) or "Posso trabalhar de outro país?"
embedder = criar_embedder()
arquivo = Path(__file__).resolve().parents[1] / f"indice-{embedder.nome.replace(':', '_').replace('/', '_')}.json"

if arquivo.exists():
    indice = Indice.carregar(arquivo)
    print(f"Índice carregado de {arquivo.name}")
else:
    t0 = time.time()
    indice = indexar(embedder)
    indice.salvar(arquivo)
    print(f"Indexados {len(indice)} trechos em {time.time() - t0:.1f}s, salvo em {arquivo.name}")

titulo(f"Pergunta: {pergunta}")
for trecho, score in indice.buscar(embedder.embed_pergunta(pergunta), k=4):
    print(f"{score:.3f}  {trecho.arquivo} > {trecho.secao}")
    print(f"       {trecho.texto[:110].replace(chr(10), ' ')}...")
