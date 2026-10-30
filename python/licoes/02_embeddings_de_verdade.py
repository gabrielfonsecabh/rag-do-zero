"""Lição 02: os mesmos pares, agora com um modelo de embedding de verdade.

O padrão é um modelo local e grátis (multilingual-e5-small, 384 números por
texto). Na primeira execução ele é baixado (cerca de 470 MB) e fica em cache.

    python licoes/02_embeddings_de_verdade.py

Para comparar com um modelo pago, sem mudar uma linha deste arquivo:

    EMBEDDINGS_PROVEDOR=openai OPENAI_API_KEY=sk-... python licoes/02_embeddings_de_verdade.py
"""
import _base  # noqa: F401
from _base import titulo

from rag import criar_embedder
from rag.provedores import EmbedderBrinquedo

pares = [
    ("Quando o notebook é trocado?", "O notebook é trocado a cada 3 anos."),
    ("Quando o notebook é trocado?", "O reembolso é pago na sexta-feira."),
    ("Quando trocam meu computador?", "O notebook é trocado a cada 3 anos."),
]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


modelo = criar_embedder()
brinquedo = EmbedderBrinquedo()
titulo(f"Brinquedo x {modelo.nome}")
for a, b in pares:
    s_b = cos(brinquedo.embed_pergunta(a), brinquedo.embed_documentos([b])[0])
    s_m = cos(modelo.embed_pergunta(a), modelo.embed_documentos([b])[0])
    print(f"brinquedo {s_b:.2f} | modelo {s_m:.2f}   {a!r} x {b!r}")

print(
    "\nO modelo dá nota alta ao terceiro par mesmo sem palavras em comum: ele\n"
    "compara significado. Repare também que as notas de um modelo real raramente\n"
    "chegam perto de 0: o que importa é a ORDEM (o certo acima do errado), não o\n"
    "valor absoluto. Por isso um limiar fixo de similaridade precisa ser calibrado\n"
    "para cada modelo."
)
