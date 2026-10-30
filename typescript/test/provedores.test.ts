import { describe, expect, it } from 'vitest'

import {
  EmbedderBrinquedo,
  EmbedderOpenAI,
  EmbedderVoyage,
  ErroDeConfiguracao,
  GeradorAnthropic,
  GeradorNenhum,
  criarEmbedder,
  criarGerador,
  type PostJson,
} from '../src/provedores.js'

describe('criarEmbedder / criarGerador', () => {
  it('geração padrão é "nenhum"', () => {
    expect(criarGerador({ env: {} })).toBeInstanceOf(GeradorNenhum)
  })

  it('troca por variável de ambiente', () => {
    expect(criarEmbedder({ env: { EMBEDDINGS_PROVEDOR: 'brinquedo' } })).toBeInstanceOf(EmbedderBrinquedo)
  })

  it('provedor inexistente lista as opções', () => {
    expect(() => criarEmbedder({ provedor: 'pinecone', env: {} })).toThrow(/Opções: brinquedo, local, openai, voyage/)
  })

  it.each([
    ['openai', 'OPENAI_API_KEY'],
    ['voyage', 'VOYAGE_API_KEY'],
  ])('embedder %s sem chave explica o que falta', (provedor, variavel) => {
    expect(() => criarEmbedder({ provedor, env: {} })).toThrow(ErroDeConfiguracao)
    expect(() => criarEmbedder({ provedor, env: {} })).toThrow(variavel)
  })

  it('gerador anthropic sem chave explica o que falta', () => {
    expect(() => criarGerador({ provedor: 'anthropic', env: {} })).toThrow('ANTHROPIC_API_KEY')
  })

  it('modelo vem da variável', () => {
    const e = criarEmbedder({ env: { EMBEDDINGS_PROVEDOR: 'openai', OPENAI_API_KEY: 'x', EMBEDDINGS_MODELO: 'text-embedding-3-large' } })
    expect(e.nome).toBe('openai:text-embedding-3-large')
  })
})

describe('chamadas às APIs pagas', () => {
  it('openai: url, cabeçalho, corpo e normalização', async () => {
    const chamadas: unknown[][] = []
    const post: PostJson = async (url, cab, corpo: any) => {
      chamadas.push([url, cab, corpo])
      return { data: corpo.input.map((_: string, index: number) => ({ index, embedding: [3, 4] })) }
    }
    const v = await new EmbedderOpenAI({ OPENAI_API_KEY: 'sk' }, undefined, post).embedDocumentos(['a', 'b'])
    expect(chamadas[0]).toEqual(['https://api.openai.com/v1/embeddings', { Authorization: 'Bearer sk' }, { model: 'text-embedding-3-small', input: ['a', 'b'] }])
    expect(v).toEqual([[0.6, 0.8], [0.6, 0.8]])
  })

  it('voyage: separa documento de pergunta', async () => {
    const tipos: string[] = []
    const post: PostJson = async (_u, _c, corpo: any) => {
      tipos.push(corpo.input_type)
      return { data: [{ index: 0, embedding: [1, 0] }] }
    }
    const e = new EmbedderVoyage({ VOYAGE_API_KEY: 'v' }, undefined, post)
    await e.embedDocumentos(['x'])
    await e.embedPergunta('y')
    expect(tipos).toEqual(['document', 'query'])
  })

  it('anthropic: monta a mensagem e junta o texto', async () => {
    let visto: any
    const post: PostJson = async (url, cab, corpo) => {
      visto = { url, cab, corpo }
      return { content: [{ type: 'text', text: 'ok [1]' }] }
    }
    const texto = await new GeradorAnthropic({ ANTHROPIC_API_KEY: 'a' }, undefined, post).gerar('sistema', 'pergunta')
    expect(texto).toBe('ok [1]')
    expect(visto.url).toBe('https://api.anthropic.com/v1/messages')
    expect(visto.cab['anthropic-version']).toBe('2023-06-01')
    expect(visto.corpo.system).toBe('sistema')
  })
})

describe('EmbedderBrinquedo', () => {
  it('determinístico, normalizado e sem acento', async () => {
    const e = new EmbedderBrinquedo()
    const [a, b] = await e.embedDocumentos(['FÉRIAS', 'ferias'])
    expect(a).toEqual(b)
    expect(a!.reduce((s, x) => s + x * x, 0)).toBeCloseTo(1, 6)
  })
})
