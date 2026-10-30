"""Lição 06: medir a busca antes de confiar nela.

Usa avaliacao/perguntas.jsonl: cada pergunta tem o arquivo onde está a resposta.
Mede se esse arquivo aparece entre os k primeiros resultados (acerto@k) e em
que posição (MRR). Compara o embedder de brinquedo com o configurado.

    python licoes/06_avaliacao.py
"""
import _base  # noqa: F401
from _base import titulo

from rag import criar_embedder, indexar
from rag.avaliacao import avaliar_busca, carregar_perguntas
from rag.provedores import EmbedderBrinquedo

perguntas = carregar_perguntas()
embedders = [EmbedderBrinquedo()]
try:
    configurado = criar_embedder()
    if configurado.nome != embedders[0].nome:
        embedders.append(configurado)
except Exception as erro:  # sem o pacote do modelo local, compara só o brinquedo
    print(f"(pulando o embedder configurado: {erro})")

for e in embedders:
    indice = indexar(e)
    titulo(e.nome)
    for k in (1, 3, 5):
        r = avaliar_busca(indice, e, perguntas, k=k)
        print(f"acerto@{k}: {r.acerto:5.0%}   MRR: {r.mrr:.2f}")
    erros = [x for x in avaliar_busca(indice, e, perguntas, k=3).resultados if x.posicao is None]
    for x in erros:
        print(f"  errou: {x.pergunta!r} (esperava {x.fonte_esperada}, veio {x.fontes_encontradas})")

print(
    "\nAcerto alto aqui não garante resposta boa, mas acerto baixo garante resposta\n"
    "ruim. Quando mudar chunking, modelo ou busca, rode isto antes e depois: é o\n"
    "único jeito de saber se a mudança melhorou alguma coisa."
)
