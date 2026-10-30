/**
 * Onde os vetores moram. Duas implementações com a mesma interface:
 *
 * - ArmazenamentoJson: um arquivo local. Zero instalação, bom para aprender.
 * - ArmazenamentoPostgres: Postgres com pgvector (Supabase serve). É o de
 *   produção, com busca híbrida (semântica + palavra-chave).
 *
 * `criarArmazenamento()` escolhe pelo DATABASE_URL: definido, usa Postgres.
 *
 * Os dois gravam o nome do embedder que indexou e recusam buscar com outro.
 * Vetores de modelos diferentes não se comparam, e sem essa trava a busca
 * devolveria lixo sem erro nenhum.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import pg from 'pg'

import type { Trecho } from './chunking.js'
import type { Vetor } from './provedores.js'

export interface Achado {
  trecho: Trecho
  score: number
}

export interface Armazenamento {
  gravar(colecao: string, embedder: string, trechos: Trecho[], vetores: Vetor[]): Promise<void>
  buscar(colecao: string, embedder: string, pergunta: string, vetor: Vetor, k: number): Promise<Achado[]>
  fechar(): Promise<void>
}

export class IndiceIncompativel extends Error {}

function conferir(colecao: string, gravado: string | undefined, pedido: string) {
  if (gravado === undefined) throw new IndiceIncompativel(`A coleção '${colecao}' não existe. Rode "npm run indexar" primeiro.`)
  if (gravado !== pedido) {
    throw new IndiceIncompativel(
      `A coleção '${colecao}' foi indexada com '${gravado}' e a busca está usando '${pedido}'. Reindexe com o modelo novo.`,
    )
  }
}

const produtoInterno = (a: Vetor, b: Vetor) => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0)

interface ArquivoJson {
  [colecao: string]: { embedder: string; trechos: Trecho[]; vetores: Vetor[] }
}

export class ArmazenamentoJson implements Armazenamento {
  constructor(private readonly caminho: string) {}

  private ler(): ArquivoJson {
    return existsSync(this.caminho) ? JSON.parse(readFileSync(this.caminho, 'utf8')) : {}
  }

  async gravar(colecao: string, embedder: string, trechos: Trecho[], vetores: Vetor[]) {
    const dados = this.ler()
    dados[colecao] = { embedder, trechos, vetores: vetores.map((v) => v.map((x) => Math.round(x * 1e6) / 1e6)) }
    writeFileSync(this.caminho, JSON.stringify(dados))
  }

  async buscar(colecao: string, embedder: string, _pergunta: string, vetor: Vetor, k: number) {
    const c = this.ler()[colecao]
    conferir(colecao, c?.embedder, embedder)
    return c!.trechos
      .map((trecho, i) => ({ trecho, score: produtoInterno(c!.vetores[i]!, vetor) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
  }

  async fechar() {}
}

const vetorSql = (v: Vetor) => `[${v.map((x) => x.toFixed(6)).join(',')}]`

export class ArmazenamentoPostgres implements Armazenamento {
  private readonly pool: pg.Pool
  constructor(url: string, private readonly hibrida = true) {
    this.pool = new pg.Pool({ connectionString: url })
  }

  /** Cria o esquema de sql/0001_rag.sql. Pode rodar de novo sem erro. */
  async criarEsquema(caminhoSql: string) {
    await this.pool.query(await readFile(caminhoSql, 'utf8'))
  }

  async gravar(colecao: string, embedder: string, trechos: Trecho[], vetores: Vetor[]) {
    const dim = vetores[0]?.length ?? 0
    const { rows } = await this.pool.query(
      `select atttypmod as dim from pg_attribute where attrelid = 'rag.trechos'::regclass and attname = 'embedding'`,
    )
    if (dim && rows[0]?.dim !== dim) {
      throw new IndiceIncompativel(
        `O modelo gera ${dim} dimensões e a tabela foi criada com ${rows[0]?.dim}. Ajuste vector(N) em sql/0001_rag.sql e recrie.`,
      )
    }
    const cliente = await this.pool.connect()
    try {
      await cliente.query('begin')
      await cliente.query('delete from rag.colecoes where id = $1', [colecao])
      await cliente.query('insert into rag.colecoes (id, embedder, dimensao) values ($1, $2, $3)', [colecao, embedder, dim])
      for (const [i, t] of trechos.entries()) {
        await cliente.query(
          'insert into rag.trechos (colecao, arquivo, secao, texto, embedding) values ($1, $2, $3, $4, $5::vector)',
          [colecao, t.arquivo, t.secao, t.texto, vetorSql(vetores[i]!)],
        )
      }
      await cliente.query('commit')
    } catch (erro) {
      await cliente.query('rollback')
      throw erro
    } finally {
      cliente.release()
    }
  }

  async buscar(colecao: string, embedder: string, pergunta: string, vetor: Vetor, k: number) {
    const { rows: c } = await this.pool.query('select embedder from rag.colecoes where id = $1', [colecao])
    conferir(colecao, c[0]?.embedder, embedder)
    const { rows } = this.hibrida
      ? await this.pool.query(
          'select arquivo, secao, texto, score from rag.buscar_hibrido($1, $2, $3::vector, $4)',
          [colecao, pergunta, vetorSql(vetor), k],
        )
      : await this.pool.query(
          'select arquivo, secao, texto, similaridade as score from rag.buscar($1, $2::vector, $3)',
          [colecao, vetorSql(vetor), k],
        )
    return rows.map((r, i) => ({
      trecho: { id: `${r.arquivo}#${i}`, arquivo: r.arquivo, secao: r.secao, texto: r.texto },
      score: Number(r.score),
    }))
  }

  async fechar() {
    await this.pool.end()
  }
}

export function criarArmazenamento(env: Record<string, string | undefined> = process.env): Armazenamento {
  return env.DATABASE_URL
    ? new ArmazenamentoPostgres(env.DATABASE_URL, env.BUSCA_HIBRIDA !== 'nao')
    : new ArmazenamentoJson(env.INDICE_JSON ?? 'indice.json')
}
