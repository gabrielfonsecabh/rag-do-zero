"""Divisão de documentos em trechos (chunks).

Regras deste chunker, cada uma com um motivo:

1. Respeita a estrutura do markdown: um trecho nunca mistura duas seções. Uma
   pergunta sobre "abono" não deve trazer junto metade da seção de "datas
   bloqueadas".
2. Cada trecho carrega o caminho da seção ("Férias > Abono"). O título é a
   melhor pista de assunto que existe, e um parágrafo solto perde essa pista.
3. Seção grande é quebrada por parágrafo e, se preciso, por frase, com
   sobreposição, para uma ideia que cruza o corte aparecer inteira em algum
   trecho.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Trecho:
    id: str
    arquivo: str
    secao: str
    texto: str
    metadados: dict = field(default_factory=dict)

    @property
    def texto_para_embedding(self) -> str:
        """O que vai para o modelo de embedding: seção + conteúdo."""
        return f"{self.secao}\n{self.texto}" if self.secao else self.texto


def _secoes(texto: str) -> list[tuple[str, str]]:
    """Divide o markdown por títulos, devolvendo (caminho_da_secao, corpo)."""
    caminho: list[str] = []
    atual: list[str] = []
    saida: list[tuple[str, str]] = []

    def fechar():
        corpo = "\n".join(atual).strip()
        if corpo:
            saida.append((" > ".join(caminho), corpo))
        atual.clear()

    for linha in texto.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", linha)
        if m:
            fechar()
            nivel = len(m.group(1))
            caminho[:] = caminho[: nivel - 1] + [m.group(2).strip()]
        else:
            atual.append(linha)
    fechar()
    return saida


def _quebrar(corpo: str, max_chars: int, sobreposicao: int) -> list[str]:
    if len(corpo) <= max_chars:
        return [corpo]
    pedacos = [p.strip() for p in re.split(r"\n\s*\n", corpo) if p.strip()]
    # parágrafo maior que o limite vira frases
    unidades: list[str] = []
    for p in pedacos:
        if len(p) <= max_chars:
            unidades.append(p)
        else:
            unidades.extend(s for s in re.split(r"(?<=[.!?])\s+", p) if s)

    trechos: list[str] = []
    atual = ""
    for u in unidades:
        if atual and len(atual) + len(u) + 2 > max_chars:
            trechos.append(atual)
            cauda = atual[-sobreposicao:] if sobreposicao else ""
            # começa a cauda numa fronteira de palavra
            cauda = cauda[cauda.find(" ") + 1 :] if " " in cauda else cauda
            atual = f"{cauda} {u}".strip() if cauda else u
        else:
            atual = f"{atual}\n\n{u}" if atual else u
    if atual:
        trechos.append(atual)
    return trechos


def dividir_markdown(
    texto: str, arquivo: str, max_chars: int = 800, sobreposicao: int = 150
) -> list[Trecho]:
    if sobreposicao >= max_chars:
        raise ValueError("a sobreposição precisa ser menor que o tamanho máximo do trecho")
    trechos: list[Trecho] = []
    for secao, corpo in _secoes(texto):
        for parte in _quebrar(corpo, max_chars, sobreposicao):
            trechos.append(Trecho(id=f"{arquivo}#{len(trechos)}", arquivo=arquivo, secao=secao, texto=parte))
    return trechos
