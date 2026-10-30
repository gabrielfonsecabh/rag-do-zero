# 06. Avaliação

Sem medir, toda mudança em RAG é palpite. "Melhorou" vira "pareceu melhor nas três perguntas que eu testei".

## Meça a busca separada da geração

O arquivo `avaliacao/perguntas.jsonl` tem perguntas com o arquivo onde está a resposta. Para cada uma, a avaliação busca e confere:

- **acerto@k:** a fonte certa apareceu entre os k primeiros?
- **MRR:** em que posição? 1º lugar vale 1, 2º vale 0,5, 4º vale 0,25, não achar vale 0. A média disso é o MRR.

```bash
python licoes/06_avaliacao.py
```

A lição compara o embedder de brinquedo com o configurado. No manual de exemplo, o brinquedo já acerta bastante porque as perguntas repetem as palavras do texto; perguntas com sinônimos mostram a diferença.

## Monte o seu conjunto de perguntas

- **Comece com 20 a 50 perguntas reais**, do jeito que as pessoas perguntam, com erro de digitação e tudo.
- **Inclua perguntas sem resposta** na base (como a do plano de saúde aqui). Elas testam se o sistema admite que não sabe.
- **Inclua perguntas difíceis de propósito:** sinônimos, siglas, números, perguntas que dependem de dois trechos.
- **Rode antes e depois de cada mudança** (tamanho de trecho, modelo, k, busca híbrida). Guarde os números.

## E a resposta final?

Depois que a busca está boa, avalie a resposta: ela está correta? cita a fonte? admite quando não sabe? Para poucas dezenas de perguntas, uma pessoa lendo é o melhor avaliador. Para centenas, dá para usar um segundo modelo como juiz, com uma rubrica clara, e conferir uma amostra à mão.
