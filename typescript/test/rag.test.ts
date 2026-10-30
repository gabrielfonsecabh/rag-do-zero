import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { beforeAll, describe, expect, it } from 'vitest'

import { ArmazenamentoJson, IndiceIncompativel } from '../src/armazenamento.js'
import { dividirMarkdown } from '../src/chunking.js'
import { EmbedderBrinquedo, GeradorNenhum } from '../src/provedores.js'
import { indexar, responder } from '../src/rag.js'

describe('dividirMarkdown', () => {
  it('um trecho por seção, com o caminho', () => {
    const t = dividirMarkdown('# Manual\n\n## A\n\nTexto A.\n\n## B\n\n### C\n\nTexto C.\n', 'm.md')
    expect(t.map((x) => x.secao)).toEqual(['Manual > A', 'Manual > B > C'])
  })

  it('seção grande quebra respeitando o limite', () => {
    const corpo = Array.from({ length: 60 }, (_, i) => `Frase número ${i} sobre o assunto.`).join(' ')
    const t = dividirMarkdown(`# T\n\n${corpo}`, 't.md', 300, 60)
    expect(t.length).toBeGreaterThan(1)
    expect(t.every((x) => x.texto.length <= 360)).toBe(true)
  })
})

describe('RAG de ponta a ponta (brinquedo + JSON)', () => {
  const embedder = new EmbedderBrinquedo()
  const armazenamento = new ArmazenamentoJson(join(mkdtempSync(join(tmpdir(), 'rag-')), 'indice.json'))
  const deps = { embedder, gerador: new GeradorNenhum(), armazenamento }

  beforeAll(async () => {
    await indexar(embedder, armazenamento)
  })

  it('acha a fonte certa e monta o prompt', async () => {
    const r = await responder('Qual o limite de gasto com alimentação numa viagem?', deps, { k: 3 })
    expect(r.trechos[0]!.trecho.arquivo).toBe('reembolso.md')
    expect(r.texto).toBeNull()
    expect(r.prompt).toContain('[1] (fonte: reembolso.md')
  })

  it('similaridade mínima corta o irrelevante', async () => {
    const r = await responder('xyzzy plugh', deps, { similaridadeMinima: 0.2 })
    expect(r.trechos).toEqual([])
    expect(r.prompt).toContain('(nenhum trecho encontrado)')
  })

  it('recusa buscar com modelo diferente do que indexou', async () => {
    const outro = { ...deps, embedder: new EmbedderBrinquedo(128) }
    await expect(responder('férias', outro)).rejects.toThrow(IndiceIncompativel)
  })
})
