"""Lição 01: o que é um embedding, construído à mão.

Um embedding é uma lista de números que representa um texto. Textos parecidos
viram listas parecidas, e "parecido" se mede pelo cosseno do ângulo entre elas.

Aqui o embedding é o mais simples possível: contar palavras. Rode e repare no
último exemplo, onde ele falha.

    python licoes/01_embedding_de_brinquedo.py
"""
import _base  # noqa: F401
from _base import titulo

from rag.provedores import EmbedderBrinquedo, palavras

e = EmbedderBrinquedo(dimensao=16)  # pequeno para dar para ver os números

titulo("1. Texto vira lista de palavras")
frase = "O notebook é trocado a cada 3 anos."
print(frase, "->", palavras(frase))

titulo("2. Lista de palavras vira vetor (16 números)")
v = e.embed_pergunta(frase)
print([round(x, 2) for x in v])


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))  # vetores já normalizados


titulo("3. Similaridade: 1 = idênticos, 0 = nada em comum")
pares = [
    ("Quando o notebook é trocado?", "O notebook é trocado a cada 3 anos."),
    ("Quando o notebook é trocado?", "O reembolso é pago na sexta-feira."),
    ("Quando trocam meu computador?", "O notebook é trocado a cada 3 anos."),
]
e = EmbedderBrinquedo()
for a, b in pares:
    print(f"{cos(e.embed_pergunta(a), e.embed_pergunta(b)):.2f}  {a!r}  x  {b!r}")

print(
    "\nRepare no terceiro par: a pergunta é a mesma ideia do primeiro, mas usa\n"
    "'computador' e 'trocam' em vez de 'notebook' e 'trocado'. Contando palavras,\n"
    "a similaridade despenca. Um modelo de embedding de verdade aprendeu que essas\n"
    "palavras querem dizer quase a mesma coisa. É a lição 02."
)
