/** npm run avaliar  Mede acerto@k e MRR da busca com avaliacao/perguntas.jsonl. */
import '../env.js'
import { readFileSync } from 'node:fs'

import { criarArmazenamento } from '../armazenamento.js'
import { criarEmbedder } from '../provedores.js'
import { ARQUIVO_PERGUNTAS } from '../rag.js'

const perguntas = readFileSync(ARQUIVO_PERGUNTAS, 'utf8')
  .split('\n')
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l) as { pergunta: string; fonte: string | null })
  .filter((p) => p.fonte)

const embedder = criarEmbedder()
const armazenamento = criarArmazenamento()
try {
  for (const k of [1, 3, 5]) {
    let acertos = 0
    let rr = 0
    for (const p of perguntas) {
      const achados = await armazenamento.buscar('manual', embedder.nome, p.pergunta, await embedder.embedPergunta(p.pergunta), k)
      const pos = achados.findIndex((a) => a.trecho.arquivo === p.fonte)
      if (pos >= 0) {
        acertos++
        rr += 1 / (pos + 1)
      }
    }
    console.log(`acerto@${k}: ${Math.round((100 * acertos) / perguntas.length)}%   MRR: ${(rr / perguntas.length).toFixed(2)}`)
  }
} finally {
  await armazenamento.fechar()
}
