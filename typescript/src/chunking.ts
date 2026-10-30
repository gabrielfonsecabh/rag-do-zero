/**
 * Divide markdown em trechos: nunca mistura duas seções, cada trecho leva o
 * caminho da seção ("Férias > Abono") e seção grande quebra por parágrafo e
 * frase, com sobreposição. Mesma lógica de python/rag/chunking.py.
 */
export interface Trecho {
  id: string
  arquivo: string
  secao: string
  texto: string
}

export const textoParaEmbedding = (t: Trecho) => (t.secao ? `${t.secao}\n${t.texto}` : t.texto)

function secoes(texto: string): Array<[string, string]> {
  const caminho: string[] = []
  let atual: string[] = []
  const saida: Array<[string, string]> = []
  const fechar = () => {
    const corpo = atual.join('\n').trim()
    if (corpo) saida.push([caminho.join(' > '), corpo])
    atual = []
  }
  for (const linha of texto.split(/\r?\n/)) {
    const m = linha.match(/^(#{1,6})\s+(.*)$/)
    if (m) {
      fechar()
      const nivel = m[1]!.length
      caminho.splice(nivel - 1, caminho.length, m[2]!.trim())
    } else {
      atual.push(linha)
    }
  }
  fechar()
  return saida
}

function quebrar(corpo: string, maxChars: number, sobreposicao: number): string[] {
  if (corpo.length <= maxChars) return [corpo]
  const unidades: string[] = []
  for (const p of corpo.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean)) {
    if (p.length <= maxChars) unidades.push(p)
    else unidades.push(...p.split(/(?<=[.!?])\s+/).filter(Boolean))
  }
  const trechos: string[] = []
  let atual = ''
  for (const u of unidades) {
    if (atual && atual.length + u.length + 2 > maxChars) {
      trechos.push(atual)
      let cauda = sobreposicao ? atual.slice(-sobreposicao) : ''
      if (cauda.includes(' ')) cauda = cauda.slice(cauda.indexOf(' ') + 1)
      atual = cauda ? `${cauda} ${u}`.trim() : u
    } else {
      atual = atual ? `${atual}\n\n${u}` : u
    }
  }
  if (atual) trechos.push(atual)
  return trechos
}

export function dividirMarkdown(texto: string, arquivo: string, maxChars = 800, sobreposicao = 150): Trecho[] {
  if (sobreposicao >= maxChars) throw new Error('a sobreposição precisa ser menor que o tamanho máximo do trecho')
  const trechos: Trecho[] = []
  for (const [secao, corpo] of secoes(texto)) {
    for (const parte of quebrar(corpo, maxChars, sobreposicao)) {
      trechos.push({ id: `${arquivo}#${trechos.length}`, arquivo, secao, texto: parte })
    }
  }
  return trechos
}
