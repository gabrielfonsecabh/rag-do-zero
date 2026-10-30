/** O RAG de ponta a ponta: indexar, buscar, montar o prompt, gerar. */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import type { Achado, Armazenamento } from './armazenamento.js'
import { dividirMarkdown, textoParaEmbedding, type Trecho } from './chunking.js'
import type { Embedder, Gerador } from './provedores.js'

const RAIZ = resolve(fileURLToPath(import.meta.url), '../../..')
export const PASTA_CORPUS = join(RAIZ, 'corpus')
export const ARQUIVO_SQL = join(RAIZ, 'sql', '0001_rag.sql')
export const ARQUIVO_PERGUNTAS = join(RAIZ, 'avaliacao', 'perguntas.jsonl')

export const SISTEMA = `Você responde perguntas usando SOMENTE os trechos de documentos fornecidos.

Regras:
- Se a resposta não estiver nos trechos, diga que não encontrou essa informação nos documentos. Não complete com conhecimento próprio.
- Cite a fonte de cada afirmação com o número do trecho entre colchetes, por exemplo [2].
- Responda em português, de forma direta, em poucas frases.`

export function carregarDocumentos(pasta = PASTA_CORPUS): Array<[string, string]> {
  const arquivos: string[] = []
  const andar = (dir: string) => {
    for (const nome of readdirSync(dir).sort()) {
      const caminho = join(dir, nome)
      if (statSync(caminho).isDirectory()) andar(caminho)
      else if (/\.(md|txt)$/i.test(nome)) arquivos.push(caminho)
    }
  }
  andar(pasta)
  if (!arquivos.length) throw new Error(`nenhum .md ou .txt em ${pasta}`)
  return arquivos.map((a) => [relative(pasta, a), readFileSync(a, 'utf8')])
}

export async function indexar(
  embedder: Embedder,
  armazenamento: Armazenamento,
  opcoes: { colecao?: string; pasta?: string; maxChars?: number; sobreposicao?: number } = {},
): Promise<Trecho[]> {
  const trechos = carregarDocumentos(opcoes.pasta).flatMap(([nome, texto]) =>
    dividirMarkdown(texto, nome, opcoes.maxChars, opcoes.sobreposicao),
  )
  const vetores = await embedder.embedDocumentos(trechos.map(textoParaEmbedding))
  await armazenamento.gravar(opcoes.colecao ?? 'manual', embedder.nome, trechos, vetores)
  return trechos
}

export function montarPrompt(pergunta: string, achados: Achado[]): string {
  const contexto = achados.length
    ? achados.map(({ trecho: t }, i) => `[${i + 1}] (fonte: ${t.arquivo}, seção: ${t.secao})\n${t.texto}`).join('\n\n')
    : '(nenhum trecho encontrado)'
  return `Trechos dos documentos:\n\n${contexto}\n\nPergunta: ${pergunta}`
}

export interface Resposta {
  pergunta: string
  trechos: Achado[]
  prompt: string
  texto: string | null
}

export async function responder(
  pergunta: string,
  deps: { embedder: Embedder; gerador: Gerador; armazenamento: Armazenamento },
  opcoes: { colecao?: string; k?: number; similaridadeMinima?: number } = {},
): Promise<Resposta> {
  const vetor = await deps.embedder.embedPergunta(pergunta)
  let trechos = await deps.armazenamento.buscar(opcoes.colecao ?? 'manual', deps.embedder.nome, pergunta, vetor, opcoes.k ?? 4)
  if (opcoes.similaridadeMinima !== undefined) trechos = trechos.filter((a) => a.score >= opcoes.similaridadeMinima!)
  const prompt = montarPrompt(pergunta, trechos)
  return { pergunta, trechos, prompt, texto: await deps.gerador.gerar(SISTEMA, prompt) }
}
