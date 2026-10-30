# RAG do zero

**Aprenda RAG construindo um.** Uma trilha em português, em 7 lições, com código que roda em cada passo: do embedding feito à mão até um RAG em produção com Postgres e pgvector.

RAG (geração aumentada por recuperação) é como se faz um modelo de linguagem responder com base nos **seus** documentos: manuais, políticas, contratos, base de conhecimento. Em vez de responder de memória, o modelo recebe os trechos certos junto com a pergunta e responde lendo, citando a fonte.

- **Grátis para começar:** os embeddings rodam no seu computador com um modelo aberto. Não precisa de cartão nem de chave de API.
- **Troca para modelo pago com uma variável:** OpenAI, Voyage ou Claude, sem mexer no código.
- **Duas versões:** lições em **Python**, que é a língua padrão de RAG, e um **app em TypeScript** com CLI e API HTTP, pronto para plugar num sistema.
- **Sem framework mágico:** cada etapa é um arquivo curto que dá para ler inteiro. Depois de entender, você escolhe se quer LangChain, LlamaIndex ou nada.

## A trilha

| # | Lição | Código |
|---|---|---|
| 01 | [O que é RAG](trilha/01-o-que-e-rag.md) | |
| 02 | [Embeddings](trilha/02-embeddings.md) | `01_embedding_de_brinquedo.py`, `02_embeddings_de_verdade.py` |
| 03 | [Chunking](trilha/03-chunking.md) | `03_chunking.py` |
| 04 | [Busca vetorial](trilha/04-busca-vetorial.md) | `04_busca_vetorial.py` |
| 05 | [Geração com contexto](trilha/05-geracao.md) | `05_rag_completo.py` |
| 06 | [Avaliação](trilha/06-avaliacao.md) | `06_avaliacao.py` |
| 07 | [Produção com Postgres e pgvector](trilha/07-producao.md) | `07_pgvector.py` e o app em `typescript/` |

A base de exemplo é o manual de uma empresa fictícia (`corpus/`), com perguntas de avaliação em `avaliacao/perguntas.jsonl`. Troque pelos seus documentos quando quiser: qualquer `.md` ou `.txt` na pasta.

## Começando (Python)

Precisa de Python 3.10 ou mais novo.

```bash
git clone https://github.com/gabrielfonsecabh/rag-do-zero && cd rag-do-zero/python
python -m venv .venv && source .venv/bin/activate   # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

python licoes/01_embedding_de_brinquedo.py
python licoes/02_embeddings_de_verdade.py      # baixa o modelo local na primeira vez (~470 MB)
python licoes/05_rag_completo.py "Quantos dias de férias posso vender?"
```

Sem chave de API, a lição 05 mostra o prompt que o modelo receberia. Com uma chave, mostra a resposta.

## Começando (TypeScript)

Precisa de Node 20 ou mais novo.

```bash
cd typescript
npm install
npm run indexar                               # cria indice.json com o corpus de exemplo
npm run perguntar -- "Posso trabalhar de outro país?"
npm run avaliar                               # acerto@k e MRR
npm run servidor                              # POST http://localhost:3000/perguntar
```

Com `DATABASE_URL` definido, o app grava e busca no Postgres com pgvector (lição 07) em vez do arquivo local.

## Trocar de modelo

Tudo passa por duas funções, `criar_embedder()` e `criar_gerador()` em Python (`criarEmbedder()` e `criarGerador()` em TypeScript). Elas leem variáveis de ambiente, então trocar de modelo não muda código. Copie `.env.example` para `.env` ou exporte no terminal.

**Embeddings** (`EMBEDDINGS_PROVEDOR`)

| Valor | O que é | Precisa de |
|---|---|---|
| `local` (padrão) | multilingual-e5-small, roda na sua máquina, grátis | nada |
| `openai` | text-embedding-3-small | `OPENAI_API_KEY` |
| `voyage` | voyage-3.5-lite | `VOYAGE_API_KEY` |
| `brinquedo` | contagem de palavras, para testes e para a lição 01 | nada |

**Geração** (`GERACAO_PROVEDOR`)

| Valor | O que é | Precisa de |
|---|---|---|
| `nenhum` (padrão) | não chama modelo, mostra o prompt montado | nada |
| `anthropic` | Claude | `ANTHROPIC_API_KEY` |
| `openai` | qualquer API no formato da OpenAI, inclusive Ollama local e grátis | `OPENAI_API_KEY` e, para Ollama, `OPENAI_BASE_URL` |

`EMBEDDINGS_MODELO` e `GERACAO_MODELO` escolhem um modelo específico do provedor.

Exemplo, trocando tudo para pago:

```bash
export EMBEDDINGS_PROVEDOR=openai OPENAI_API_KEY=sk-...
export GERACAO_PROVEDOR=anthropic ANTHROPIC_API_KEY=sk-ant-...
python licoes/05_rag_completo.py "Qual o limite de hospedagem?"
```

**Ao trocar o modelo de embeddings, reindexe.** Vetores de modelos diferentes não se comparam, e uma busca que mistura os dois devolve resultado ruim sem dar erro. Por isso o índice grava qual modelo o criou e recusa buscar com outro, com uma mensagem dizendo o que fazer.

**Para acrescentar um provedor**, escreva uma classe com os mesmos métodos (`embed_documentos`, `embed_pergunta` ou `gerar`) em `python/rag/provedores.py` ou `typescript/src/provedores.ts`, e registre no dicionário `_EMBEDDERS` / `EMBEDDERS` ou `_GERADORES` / `GERADORES`.

## Estrutura

```
corpus/          documentos de exemplo (empresa fictícia)
avaliacao/       perguntas com a fonte esperada
trilha/          as 7 lições em texto
sql/             esquema do Postgres + pgvector, com busca híbrida
python/
  rag/           o pacote: provedores, chunking, índice, pipeline, avaliação, postgres
  licoes/        um script executável por lição
  tests/         pytest
typescript/
  src/           provedores, chunking, armazenamento (JSON e Postgres), RAG, CLI e servidor HTTP
  test/          vitest
```

## Testes

```bash
cd python && pip install -r requirements-dev.txt && pytest
cd typescript && npm test && npm run typecheck
```

Os testes usam o embedder de brinquedo e chamadas simuladas às APIs, então rodam sem internet e sem chave.

## Contribuindo

Pull requests são bem-vindos, principalmente: novos provedores, exemplos com PDF, re-ranking, e traduções da trilha. Toda mudança na busca vem com o resultado da avaliação antes e depois.

## Licença

[MIT](LICENSE). Use, adapte e ensine, inclusive comercialmente.

Criado por [Gabriel Fonseca](https://github.com/gabrielfonsecabh).
