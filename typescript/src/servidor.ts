/**
 * API HTTP mínima, sem framework, para plugar o RAG num app.
 *
 *   npm run servidor
 *   curl -s localhost:3000/perguntar -H 'content-type: application/json' \
 *        -d '{"pergunta":"Posso trabalhar de outro país?"}'
 */
import './env.js'
import { createServer } from 'node:http'

import { criarArmazenamento } from './armazenamento.js'
import { criarEmbedder, criarGerador } from './provedores.js'
import { responder } from './rag.js'

const deps = { embedder: criarEmbedder(), gerador: criarGerador(), armazenamento: criarArmazenamento() }
const porta = Number(process.env.PORTA ?? 3000)

const servidor = createServer(async (req, res) => {
  const enviar = (status: number, corpo: unknown) => {
    res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' })
    res.end(JSON.stringify(corpo))
  }
  if (req.method === 'GET' && req.url === '/saude') return enviar(200, { ok: true, embedder: deps.embedder.nome, gerador: deps.gerador.nome })
  if (req.method !== 'POST' || req.url !== '/perguntar') return enviar(404, { erro: 'use POST /perguntar ou GET /saude' })

  let corpo = ''
  for await (const pedaco of req) {
    corpo += pedaco
    if (corpo.length > 10_000) return enviar(413, { erro: 'pergunta grande demais' })
  }
  try {
    const { pergunta, k } = JSON.parse(corpo || '{}')
    if (typeof pergunta !== 'string' || !pergunta.trim()) return enviar(400, { erro: 'envie {"pergunta": "..."}' })
    const r = await responder(pergunta, deps, { k: Math.min(Number(k) || 4, 10) })
    enviar(200, {
      resposta: r.texto,
      fontes: r.trechos.map(({ trecho, score }) => ({ arquivo: trecho.arquivo, secao: trecho.secao, score, texto: trecho.texto })),
      prompt: r.texto === null ? r.prompt : undefined,
    })
  } catch (erro) {
    enviar(500, { erro: erro instanceof Error ? erro.message : String(erro) })
  }
})

servidor.listen(porta, () => console.log(`RAG ouvindo em http://localhost:${porta} (${deps.embedder.nome}, ${deps.gerador.nome})`))
