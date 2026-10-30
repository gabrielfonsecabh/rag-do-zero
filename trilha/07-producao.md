# 07. Produção com Postgres e pgvector

O índice em JSON serve para aprender. Em produção, com muitos documentos, várias pessoas consultando e documentos mudando, o lugar dos vetores é o banco de dados. O Postgres com a extensão **pgvector** faz isso bem e já vem pronto no Supabase.

## O esquema

Está em `sql/0001_rag.sql`, usado pelas versões Python e TypeScript:

- `rag.colecoes`: cada conjunto de documentos e **o embedder que o indexou**.
- `rag.trechos`: o texto, a seção, o vetor (`vector(384)`) e uma coluna de busca por palavra-chave em português (`tsvector`).
- Índice **HNSW** no vetor (busca aproximada rápida) e **GIN** no texto.
- `rag.buscar(...)`: busca semântica.
- `rag.buscar_hibrido(...)`: semântica + palavra-chave, combinadas por **Reciprocal Rank Fusion**.

**Antes de rodar:** a dimensão do vetor é fixa porque o índice HNSW exige. 384 é a do modelo local padrão. Com outro modelo, troque `384` em todo o arquivo (OpenAI `text-embedding-3-small`: 1536; Voyage `voyage-3.5-lite`: 1024).

## Rodando

Python:

```bash
pip install 'psycopg[binary]'
export DATABASE_URL=postgresql://usuario:senha@host:5432/banco
python licoes/07_pgvector.py indexar
python licoes/07_pgvector.py buscar "Qual o limite de hospedagem?"
```

TypeScript (com `DATABASE_URL` definido, o app usa o Postgres sozinho):

```bash
cd typescript
npm run indexar
npm run perguntar -- "Qual o limite de hospedagem?"
npm run servidor
```

## Por que busca híbrida

A busca semântica entende sinônimos, mas se confunde com termos exatos: o código de um produto, uma sigla, um número de artigo de lei. A busca por palavra-chave é o contrário. A fusão RRF dá a cada trecho 1 / (60 + posição) em cada lista e soma as notas. Um trecho que vai bem nas duas sobe, e o resultado herda o melhor de cada uma.

## Lista para colocar no ar

- **Atualização:** ao mudar um documento, apague os trechos dele e indexe de novo. Guarde uma data ou um hash do arquivo para saber o que mudou.
- **Permissão:** se nem todo mundo pode ver todo documento, filtre na busca (por exemplo uma coluna de empresa ou de área com Row Level Security no Supabase). Nunca confie no prompt para esconder informação.
- **Custos:** com API paga, o custo de indexar é pago uma vez; o de perguntar, a cada pergunta. Guarde em cache as perguntas repetidas.
- **Registro:** grave a pergunta, os trechos que vieram e a resposta. É assim que você descobre as perguntas que o sistema erra e aumenta o conjunto de avaliação.
