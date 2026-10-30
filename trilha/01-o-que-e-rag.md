# 01. O que é RAG

Um modelo de linguagem sabe o que estava nos dados de treino. Ele não conhece o manual da sua empresa, os contratos do seu cliente nem a política que mudou semana passada. E quando não sabe, muitas vezes inventa com a mesma confiança de quando sabe.

**RAG** (Retrieval-Augmented Generation, ou geração aumentada por recuperação) resolve isso sem treinar modelo nenhum: antes de perguntar ao modelo, você **busca** os trechos dos seus documentos que respondem a pergunta e **entrega junto**. O modelo deixa de responder de memória e passa a responder lendo.

## O fluxo

```
            INDEXAR (uma vez)                              PERGUNTAR (a cada pergunta)

documentos ──> trechos ──> vetores ──> índice      pergunta ──> vetor ──> busca no índice
                                                                              │
                                                                              ▼
                                                   resposta <── modelo <── prompt com os trechos
```

1. **Dividir** os documentos em trechos (lição 03).
2. **Transformar** cada trecho num vetor, uma lista de números que representa o significado (lição 02).
3. **Guardar** os vetores num índice (lição 04, e lição 07 para produção).
4. Na pergunta, **transformar** a pergunta em vetor e **buscar** os trechos mais próximos (lição 04).
5. **Montar** um prompt com os trechos e pedir a resposta ao modelo (lição 05).
6. **Medir** se a busca traz o trecho certo (lição 06).

## Quando RAG é a escolha certa

- A informação muda com frequência (políticas, preços, estoque, documentação).
- A resposta precisa citar a fonte, para alguém conferir.
- Os documentos são seus e não estão na internet.

## Quando não é

- A pergunta exige raciocinar sobre a base inteira de uma vez ("qual o tema mais comum em 10 mil chamados?"). Isso é análise de dados, não busca de trechos.
- A base é pequena o bastante para caber inteira no prompt. Aí é mais simples mandar tudo.
- O problema é o modelo não seguir um formato ou um estilo. Isso se resolve com prompt ou ajuste fino, não com busca.

## A regra que guia o resto da trilha

**Quase todo erro de RAG é erro de busca.** Se o trecho certo não chega ao prompt, nenhum modelo acerta. Por isso este repositório deixa você ver o prompt montado (lição 05) e medir a busca separada da geração (lição 06).
