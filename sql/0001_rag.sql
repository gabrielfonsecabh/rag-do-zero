-- 0001: esquema do RAG no Postgres com pgvector (Supabase ou qualquer Postgres 15+).
--
-- A dimensão do vetor é fixa porque o índice HNSW exige. 384 é a do modelo
-- local padrão (multilingual-e5-small). Usando outro modelo, troque 384 em
-- TODOS os lugares deste arquivo antes de rodar:
--   openai text-embedding-3-small: 1536 | voyage-3.5-lite: 1024
-- Pode rodar de novo sem erro.

create extension if not exists vector;

create schema if not exists rag;

-- Uma coleção é um conjunto de documentos indexado por UM embedder. O nome
-- do embedder fica gravado para a busca recusar um modelo diferente.
create table if not exists rag.colecoes (
  id         text primary key,
  embedder   text not null,
  dimensao   int  not null,
  criada_em  timestamptz not null default now()
);

create table if not exists rag.trechos (
  id         bigint generated always as identity primary key,
  colecao    text  not null references rag.colecoes(id) on delete cascade,
  arquivo    text  not null,
  secao      text  not null default '',
  texto      text  not null,
  metadados  jsonb not null default '{}'::jsonb,
  embedding  vector(384) not null,
  -- busca por palavra-chave, para a busca híbrida
  fts        tsvector generated always as (to_tsvector('portuguese', secao || ' ' || texto)) stored
);

create index if not exists trechos_colecao_idx on rag.trechos (colecao);
create index if not exists trechos_fts_idx on rag.trechos using gin (fts);
create index if not exists trechos_embedding_idx on rag.trechos
  using hnsw (embedding vector_cosine_ops);

-- Busca semântica: os k trechos de menor distância de cosseno.
create or replace function rag.buscar(
  p_colecao   text,
  p_embedding vector(384),
  p_k         int default 4
)
returns table (id bigint, arquivo text, secao text, texto text, similaridade float)
language sql stable
as $$
  select t.id, t.arquivo, t.secao, t.texto, 1 - (t.embedding <=> p_embedding)
    from rag.trechos t
   where t.colecao = p_colecao
   order by t.embedding <=> p_embedding
   limit p_k;
$$;

-- Busca híbrida: junta a semântica com a de palavra-chave por Reciprocal Rank
-- Fusion (RRF). Cada lista dá 1 / (p_rrf_k + posição) a cada trecho, e as
-- notas somam. Pega o melhor dos dois: sinônimos (semântica) e termos exatos
-- como códigos, siglas e números (palavra-chave).
create or replace function rag.buscar_hibrido(
  p_colecao   text,
  p_texto     text,
  p_embedding vector(384),
  p_k         int default 4,
  p_rrf_k     int default 60
)
returns table (id bigint, arquivo text, secao text, texto text, score float)
language sql stable
as $$
  with semantica as (
    select t.id, row_number() over (order by t.embedding <=> p_embedding) as pos
      from rag.trechos t
     where t.colecao = p_colecao
     order by t.embedding <=> p_embedding
     limit p_k * 4
  ),
  consulta as (
    select websearch_to_tsquery('portuguese', p_texto) as q
  ),
  lexica as (
    select t.id, row_number() over (order by ts_rank_cd(t.fts, c.q) desc) as pos
      from rag.trechos t, consulta c
     where t.colecao = p_colecao and t.fts @@ c.q
     order by ts_rank_cd(t.fts, c.q) desc
     limit p_k * 4
  )
  select t.id, t.arquivo, t.secao, t.texto,
         coalesce(1.0 / (p_rrf_k + s.pos), 0) + coalesce(1.0 / (p_rrf_k + l.pos), 0) as score
    from semantica s
    full outer join lexica l on l.id = s.id
    join rag.trechos t on t.id = coalesce(s.id, l.id)
   order by score desc
   limit p_k;
$$;
