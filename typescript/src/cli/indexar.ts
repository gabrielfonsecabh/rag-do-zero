/** npm run indexar [-- pasta]  Indexa a pasta (padrão: corpus/) no armazenamento configurado. */
import '../env.js'
import { ArmazenamentoPostgres, criarArmazenamento } from '../armazenamento.js'
import { criarEmbedder } from '../provedores.js'
import { ARQUIVO_SQL, PASTA_CORPUS, indexar } from '../rag.js'

const pasta = process.argv[2] ?? PASTA_CORPUS
const embedder = criarEmbedder()
const armazenamento = criarArmazenamento()
try {
  if (armazenamento instanceof ArmazenamentoPostgres) await armazenamento.criarEsquema(ARQUIVO_SQL)
  const inicio = Date.now()
  const trechos = await indexar(embedder, armazenamento, { pasta })
  const onde = process.env.DATABASE_URL ? 'Postgres' : process.env.INDICE_JSON ?? 'indice.json'
  console.log(`${trechos.length} trechos indexados com ${embedder.nome} em ${((Date.now() - inicio) / 1000).toFixed(1)}s (${onde})`)
} finally {
  await armazenamento.fechar()
}
