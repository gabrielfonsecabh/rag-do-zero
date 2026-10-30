/**
 * Provedores de embeddings e de geração de texto.
 *
 * O resto do app fala só com `Embedder` e `Gerador`. Para trocar de modelo,
 * local ou pago, muda-se uma variável de ambiente:
 *
 *   EMBEDDINGS_PROVEDOR = local | openai | voyage | brinquedo
 *   EMBEDDINGS_MODELO   = (opcional) nome do modelo
 *   GERACAO_PROVEDOR    = nenhum | anthropic | openai
 *   GERACAO_MODELO      = (opcional) nome do modelo
 *
 * Para acrescentar um provedor, escreva uma classe com os mesmos métodos e
 * registre em `EMBEDDERS` ou `GERADORES`.
 */
import { createHash } from 'node:crypto'

export type Vetor = number[]

export interface Embedder {
  /** "provedor:modelo". Fica gravado no índice para detectar troca de modelo. */
  readonly nome: string
  embedDocumentos(textos: string[]): Promise<Vetor[]>
  embedPergunta(texto: string): Promise<Vetor>
}

export interface Gerador {
  readonly nome: string
  /** null quando o provedor é "nenhum": o app mostra o prompt em vez da resposta. */
  gerar(sistema: string, mensagem: string): Promise<string | null>
}

export class ErroDeConfiguracao extends Error {}

type Env = Record<string, string | undefined>

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

export function normalizar(v: number[]): Vetor {
  const norma = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
  return norma > 0 ? v.map((x) => x / norma) : [...v]
}

function chave(env: Env, variavel: string, provedor: string): string {
  const valor = env[variavel]?.trim()
  if (!valor) {
    throw new ErroDeConfiguracao(
      `O provedor '${provedor}' precisa da variável ${variavel}. Defina no .env ou no terminal.`,
    )
  }
  return valor
}

export type PostJson = (url: string, cabecalhos: Record<string, string>, corpo: unknown) => Promise<any>

export const postJson: PostJson = async (url, cabecalhos, corpo) => {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...cabecalhos },
    body: JSON.stringify(corpo),
  })
  if (!r.ok) throw new Error(`${url} respondeu ${r.status}: ${(await r.text()).slice(0, 500)}`)
  return r.json()
}

function* emLotes<T>(itens: T[], tamanho: number) {
  for (let i = 0; i < itens.length; i += tamanho) yield itens.slice(i, i + tamanho)
}

// ---------------------------------------------------------------------------
// Embedders
// ---------------------------------------------------------------------------

const PALAVRAS_VAZIAS = new Set(
  'a o as os de da do das dos e é em no na nos nas um uma uns umas por para com sem que se ao à às ou mais menos já até como qual quais quando onde quem meu minha'.split(' '),
)

export function palavras(texto: string): string[] {
  const t = texto.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '')
  return (t.match(/[a-z0-9]+/g) ?? []).filter((p) => p.length > 1 && !PALAVRAS_VAZIAS.has(p))
}

/** Conta palavras num vetor fixo. Sem sinônimo nem contexto: serve para testes e como linha de base. */
export class EmbedderBrinquedo implements Embedder {
  readonly nome: string
  constructor(private readonly dimensao = 512) {
    this.nome = `brinquedo:bag-of-words-${dimensao}`
  }
  private vetor(texto: string): Vetor {
    const v = new Array<number>(this.dimensao).fill(0)
    for (const p of palavras(texto)) {
      const i = parseInt(createHash('md5').update(p.slice(0, 6)).digest('hex').slice(-8), 16) % this.dimensao
      v[i] = (v[i] ?? 0) + 1
    }
    return normalizar(v)
  }
  async embedDocumentos(textos: string[]) {
    return textos.map((t) => this.vetor(t))
  }
  async embedPergunta(texto: string) {
    return this.vetor(texto)
  }
}

/**
 * Modelo local e grátis via transformers.js (roda em Node, sem Python).
 * Padrão: multilingual-e5-small, 384 dimensões. A família e5 espera os
 * prefixos "query: " e "passage: "; sem eles a busca piora sem dar erro.
 */
export class EmbedderLocal implements Embedder {
  static readonly PADRAO = 'Xenova/multilingual-e5-small'
  readonly nome: string
  private extrator: Promise<any> | null = null
  private readonly e5: boolean

  constructor(private readonly modelo = EmbedderLocal.PADRAO) {
    this.nome = `local:${modelo}`
    this.e5 = modelo.toLowerCase().includes('e5')
  }

  private async encode(textos: string[]): Promise<Vetor[]> {
    if (!this.extrator) {
      this.extrator = import('@huggingface/transformers').then(({ pipeline }) =>
        pipeline('feature-extraction', this.modelo),
      )
    }
    const extrator = await this.extrator
    const saida: Vetor[] = []
    for (const lote of emLotes(textos, 32)) {
      const tensor = await extrator(lote, { pooling: 'mean', normalize: true })
      saida.push(...(tensor.tolist() as number[][]))
    }
    return saida
  }

  embedDocumentos(textos: string[]) {
    return this.encode(this.e5 ? textos.map((t) => `passage: ${t}`) : textos)
  }
  async embedPergunta(texto: string) {
    const [v] = await this.encode([this.e5 ? `query: ${texto}` : texto])
    return v!
  }
}

/** Embeddings da OpenAI (pago). OPENAI_BASE_URL aponta para qualquer serviço compatível. */
export class EmbedderOpenAI implements Embedder {
  static readonly PADRAO = 'text-embedding-3-small'
  readonly nome: string
  private readonly chave: string
  private readonly url: string
  constructor(env: Env, private readonly modelo = EmbedderOpenAI.PADRAO, private readonly post: PostJson = postJson) {
    this.chave = chave(env, 'OPENAI_API_KEY', 'openai')
    this.url = (env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1').replace(/\/$/, '')
    this.nome = `openai:${modelo}`
  }
  async embedDocumentos(textos: string[]) {
    const saida: Vetor[] = []
    for (const lote of emLotes(textos, 100)) {
      const r = await this.post(`${this.url}/embeddings`, { Authorization: `Bearer ${this.chave}` }, { model: this.modelo, input: lote })
      const dados = [...r.data].sort((a: any, b: any) => a.index - b.index)
      saida.push(...dados.map((d: any) => normalizar(d.embedding)))
    }
    return saida
  }
  async embedPergunta(texto: string) {
    const [v] = await this.embedDocumentos([texto])
    return v!
  }
}

/** Embeddings da Voyage AI (pago). input_type separa documento de pergunta. */
export class EmbedderVoyage implements Embedder {
  static readonly PADRAO = 'voyage-3.5-lite'
  readonly nome: string
  private readonly chave: string
  constructor(env: Env, private readonly modelo = EmbedderVoyage.PADRAO, private readonly post: PostJson = postJson) {
    this.chave = chave(env, 'VOYAGE_API_KEY', 'voyage')
    this.nome = `voyage:${modelo}`
  }
  private async embed(textos: string[], tipo: 'document' | 'query') {
    const saida: Vetor[] = []
    for (const lote of emLotes(textos, 100)) {
      const r = await this.post(
        'https://api.voyageai.com/v1/embeddings',
        { Authorization: `Bearer ${this.chave}` },
        { model: this.modelo, input: lote, input_type: tipo },
      )
      const dados = [...r.data].sort((a: any, b: any) => a.index - b.index)
      saida.push(...dados.map((d: any) => normalizar(d.embedding)))
    }
    return saida
  }
  embedDocumentos(textos: string[]) {
    return this.embed(textos, 'document')
  }
  async embedPergunta(texto: string) {
    const [v] = await this.embed([texto], 'query')
    return v!
  }
}

const EMBEDDERS: Record<string, (env: Env, modelo?: string) => Embedder> = {
  brinquedo: () => new EmbedderBrinquedo(),
  local: (_env, modelo) => new EmbedderLocal(modelo),
  openai: (env, modelo) => new EmbedderOpenAI(env, modelo),
  voyage: (env, modelo) => new EmbedderVoyage(env, modelo),
}

/**
 * A função de troca. Sem argumentos, lê EMBEDDINGS_PROVEDOR e EMBEDDINGS_MODELO.
 *
 *   criarEmbedder()                        // local, grátis
 *   criarEmbedder({ provedor: 'openai' })  // precisa de OPENAI_API_KEY
 */
export function criarEmbedder(opcoes: { provedor?: string; modelo?: string; env?: Env } = {}): Embedder {
  const env = opcoes.env ?? process.env
  const provedor = (opcoes.provedor ?? env.EMBEDDINGS_PROVEDOR ?? 'local').trim().toLowerCase()
  const modelo = opcoes.modelo ?? (env.EMBEDDINGS_MODELO || undefined)
  const fabrica = EMBEDDERS[provedor]
  if (!fabrica) {
    throw new ErroDeConfiguracao(`Provedor de embeddings '${provedor}' não existe. Opções: ${Object.keys(EMBEDDERS).join(', ')}`)
  }
  return fabrica(env, modelo)
}

// ---------------------------------------------------------------------------
// Geradores
// ---------------------------------------------------------------------------

export class GeradorNenhum implements Gerador {
  readonly nome = 'nenhum'
  async gerar() {
    return null
  }
}

export class GeradorAnthropic implements Gerador {
  static readonly PADRAO = 'claude-haiku-5-5'
  readonly nome: string
  private readonly chave: string
  constructor(env: Env, private readonly modelo = GeradorAnthropic.PADRAO, private readonly post: PostJson = postJson) {
    this.chave = chave(env, 'ANTHROPIC_API_KEY', 'anthropic')
    this.nome = `anthropic:${modelo}`
  }
  async gerar(sistema: string, mensagem: string) {
    const r = await this.post(
      'https://api.anthropic.com/v1/messages',
      { 'x-api-key': this.chave, 'anthropic-version': '2023-06-01' },
      { model: this.modelo, max_tokens: 1024, system: sistema, messages: [{ role: 'user', content: mensagem }] },
    )
    return (r.content ?? []).filter((b: any) => b.type === 'text').map((b: any) => b.text).join('')
  }
}

/** Formato OpenAI (chat/completions). Com OPENAI_BASE_URL serve para Ollama e outros, inclusive locais. */
export class GeradorOpenAI implements Gerador {
  static readonly PADRAO = 'gpt-4o-mini'
  readonly nome: string
  private readonly chave: string
  private readonly url: string
  constructor(env: Env, private readonly modelo = GeradorOpenAI.PADRAO, private readonly post: PostJson = postJson) {
    this.chave = chave(env, 'OPENAI_API_KEY', 'openai')
    this.url = (env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1').replace(/\/$/, '')
    this.nome = `openai:${modelo}`
  }
  async gerar(sistema: string, mensagem: string) {
    const r = await this.post(
      `${this.url}/chat/completions`,
      { Authorization: `Bearer ${this.chave}` },
      { model: this.modelo, messages: [{ role: 'system', content: sistema }, { role: 'user', content: mensagem }] },
    )
    return r.choices[0].message.content as string
  }
}

const GERADORES: Record<string, (env: Env, modelo?: string) => Gerador> = {
  nenhum: () => new GeradorNenhum(),
  anthropic: (env, modelo) => new GeradorAnthropic(env, modelo),
  openai: (env, modelo) => new GeradorOpenAI(env, modelo),
}

/** A função de troca da geração. Sem argumentos, lê GERACAO_PROVEDOR e GERACAO_MODELO. */
export function criarGerador(opcoes: { provedor?: string; modelo?: string; env?: Env } = {}): Gerador {
  const env = opcoes.env ?? process.env
  const provedor = (opcoes.provedor ?? env.GERACAO_PROVEDOR ?? 'nenhum').trim().toLowerCase()
  const modelo = opcoes.modelo ?? (env.GERACAO_MODELO || undefined)
  const fabrica = GERADORES[provedor]
  if (!fabrica) {
    throw new ErroDeConfiguracao(`Provedor de geração '${provedor}' não existe. Opções: ${Object.keys(GERADORES).join(', ')}`)
  }
  return fabrica(env, modelo)
}
