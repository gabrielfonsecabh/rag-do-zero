/** npm run perguntar -- "sua pergunta"  Responde usando o índice já criado. */
import '../env.js'
import { criarArmazenamento } from '../armazenamento.js'
import { criarEmbedder, criarGerador } from '../provedores.js'
import { responder } from '../rag.js'

const pergunta = process.argv.slice(2).join(' ') || 'Quantos dias de férias posso vender?'
const deps = { embedder: criarEmbedder(), gerador: criarGerador(), armazenamento: criarArmazenamento() }
try {
  const r = await responder(pergunta, deps)
  console.log(`\nPergunta: ${pergunta}\n\nTrechos recuperados:`)
  r.trechos.forEach(({ trecho, score }, i) => console.log(`[${i + 1}] ${score.toFixed(3)}  ${trecho.arquivo} > ${trecho.secao}`))
  if (r.texto === null) console.log(`\nPrompt que o modelo receberia (GERACAO_PROVEDOR=nenhum):\n\n${r.prompt}`)
  else console.log(`\nResposta (${deps.gerador.nome}):\n${r.texto}`)
} finally {
  await deps.armazenamento.fechar()
}
