/** Lê o .env (pasta atual ou raiz do repositório) sem dependência. O terminal vence o arquivo. */
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const raiz = resolve(fileURLToPath(import.meta.url), '../../..')
for (const caminho of [join(process.cwd(), '.env'), join(raiz, 'typescript', '.env'), join(raiz, '.env')]) {
  if (!existsSync(caminho)) continue
  for (const linha of readFileSync(caminho, 'utf8').split(/\r?\n/)) {
    const m = linha.trim().match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/)
    if (m && process.env[m[1]!] === undefined) process.env[m[1]!] = m[2]!.replace(/^['"]|['"]$/g, '')
  }
}
