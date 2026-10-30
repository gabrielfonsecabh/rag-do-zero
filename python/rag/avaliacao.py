"""Avaliação da busca: o trecho certo aparece entre os primeiros?

Duas métricas, as mais usadas para medir recuperação:

- acerto@k: fração das perguntas em que a fonte certa está entre os k
  primeiros resultados.
- MRR (mean reciprocal rank): média de 1/posição da primeira fonte certa.
  Acertar em 1º vale 1, em 2º vale 0,5, em 4º vale 0,25, não achar vale 0.

Medir a busca separada da geração é o que permite saber ONDE o RAG erra: se a
busca não traz o trecho, nenhum modelo vai acertar a resposta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .indice import Indice
from .provedores import Embedder

ARQUIVO_PERGUNTAS = Path(__file__).resolve().parents[2] / "avaliacao" / "perguntas.jsonl"


@dataclass
class ResultadoDaPergunta:
    pergunta: str
    fonte_esperada: str
    fontes_encontradas: list[str]
    posicao: int | None  # 1 = primeiro lugar; None = não apareceu


@dataclass
class Relatorio:
    k: int
    acerto: float
    mrr: float
    resultados: list[ResultadoDaPergunta]


def carregar_perguntas(caminho: str | Path = ARQUIVO_PERGUNTAS) -> list[dict]:
    linhas = Path(caminho).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in linhas if l.strip()]


def avaliar_busca(indice: Indice, embedder: Embedder, perguntas: list[dict], k: int = 3) -> Relatorio:
    indice.conferir(embedder)
    # perguntas sem fonte (a resposta certa é "não sei") não medem busca
    validas = [p for p in perguntas if p.get("fonte")]
    resultados: list[ResultadoDaPergunta] = []
    for p in validas:
        achados = indice.buscar(embedder.embed_pergunta(p["pergunta"]), k=k)
        fontes = [t.arquivo for t, _ in achados]
        posicao = fontes.index(p["fonte"]) + 1 if p["fonte"] in fontes else None
        resultados.append(ResultadoDaPergunta(p["pergunta"], p["fonte"], fontes, posicao))
    n = len(resultados) or 1
    acerto = sum(r.posicao is not None for r in resultados) / n
    mrr = sum(1 / r.posicao for r in resultados if r.posicao) / n
    return Relatorio(k, acerto, mrr, resultados)
