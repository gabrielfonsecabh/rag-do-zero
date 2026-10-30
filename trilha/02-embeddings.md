# 02. Embeddings

Um **embedding** é uma lista de números que representa um texto. O modelo de embedding é treinado para que textos com significado parecido virem listas parecidas.

"Parecido" se mede pelo **cosseno** do ângulo entre os dois vetores: 1 é a mesma direção, 0 é nada em comum. Com os vetores normalizados (comprimento 1, como todos neste projeto), o cosseno é só a soma dos produtos dos números, e é por isso que a busca é tão rápida.

## Do brinquedo ao modelo de verdade

Rode `python licoes/01_embedding_de_brinquedo.py`. Ali o embedding é feito à mão: cada palavra soma 1 numa posição do vetor. Funciona quando a pergunta usa as mesmas palavras do documento, e falha quando usa sinônimos: "computador" e "notebook" são, para ele, coisas sem relação.

Rode `python licoes/02_embeddings_de_verdade.py`. O modelo real (multilingual-e5-small) dá nota alta para "Quando trocam meu computador?" contra "O notebook é trocado a cada 3 anos", sem nenhuma palavra em comum. Ele compara significado.

## Escolhendo o modelo

| | Local (padrão) | Pago (API) |
|---|---|---|
| Custo | zero | por token, normalmente centavos por milhão |
| Privacidade | o texto não sai da máquina | o texto vai para o provedor |
| Qualidade | boa, e ótima em português nos modelos multilíngues | um pouco melhor nos casos difíceis |
| Velocidade | depende da sua CPU ou GPU | depende da rede |

Este projeto começa local e troca para pago com uma variável de ambiente (`EMBEDDINGS_PROVEDOR`). Veja "Trocar de modelo" no README.

## Três armadilhas

1. **Prefixos.** Os modelos e5 esperam `query: ` antes da pergunta e `passage: ` antes do documento. Sem eles a qualidade cai e nada avisa. O `EmbedderLocal` já faz isso. A Voyage resolve o mesmo problema com o campo `input_type`.
2. **Trocar de modelo exige reindexar.** Vetores de modelos diferentes não se comparam, mesmo que tenham o mesmo tamanho. O índice deste projeto grava o nome do modelo e recusa buscar com outro.
3. **O valor absoluto da nota engana.** Cada modelo tem a sua escala: um 0,80 pode ser ótimo num e medíocre noutro. O que importa é a ordem. Limiar fixo de similaridade precisa ser calibrado com a avaliação da lição 06.
