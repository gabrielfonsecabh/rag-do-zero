# 04. Busca vetorial

**Indexar** é calcular o vetor de cada trecho uma vez e guardar. **Buscar** é calcular o vetor da pergunta e pegar os trechos mais próximos.

Com poucos milhares de trechos, comparar a pergunta com todos é instantâneo. O índice deste projeto (`python/rag/indice.py`) faz exatamente isso: uma multiplicação de matriz e uma ordenação.

```bash
python licoes/04_busca_vetorial.py "Posso trabalhar de outro país?"
EMBEDDINGS_PROVEDOR=brinquedo python licoes/04_busca_vetorial.py "Posso trabalhar de outro país?"
```

Compare os dois resultados: é a diferença entre contar palavras e entender significado, agora na busca.

## Quantos trechos trazer (k)

- **Poucos** (1 ou 2): risco de deixar de fora o trecho que tinha a resposta.
- **Muitos** (10 ou mais): o prompt fica caro e o modelo se distrai com trecho irrelevante.
- **Ponto de partida:** de 3 a 5. Use a lição 06 para ver em que k o acerto para de subir.

## Quando a busca precisa crescer

| Situação | Solução |
|---|---|
| Dezenas de milhares de trechos | índice aproximado (HNSW) no Postgres com pgvector, lição 07 |
| Perguntas com código, sigla, número exato | busca híbrida, semântica + palavra-chave, lição 07 |
| Os 10 primeiros estão bons mas a ordem não | re-ranking: um segundo modelo reordena os candidatos |
| Pergunta vaga ou com vários assuntos | reescrever a pergunta, ou dividir em subperguntas, antes de buscar |
