"""Índice vetorial em memória, salvo em JSON.

Pequeno de propósito: dá para abrir o arquivo e ver os vetores. Para milhares
de documentos ou acesso concorrente, use o Postgres com pgvector (lição 07).

O índice grava qual embedder o criou. Buscar com um modelo diferente do que
indexou não dá erro nenhum por conta própria: só devolve resultados ruins,
porque vetores de modelos diferentes não são comparáveis. Por isso a busca
confere e falha alto.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunking import Trecho
from .provedores import Embedder


class IndiceIncompativel(RuntimeError):
    pass


class Indice:
    def __init__(self, embedder_nome: str):
        self.embedder_nome = embedder_nome
        self.trechos: list[Trecho] = []
        self._matriz: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.trechos)

    @property
    def dimensao(self) -> int:
        return 0 if self._matriz is None else int(self._matriz.shape[1])

    def adicionar(self, trechos: list[Trecho], vetores: list[list[float]]) -> None:
        if len(trechos) != len(vetores):
            raise ValueError("cada trecho precisa de exatamente um vetor")
        if not trechos:
            return
        novos = np.asarray(vetores, dtype=np.float32)
        if self._matriz is not None and novos.shape[1] != self._matriz.shape[1]:
            raise IndiceIncompativel(
                f"vetores com {novos.shape[1]} dimensões num índice de {self._matriz.shape[1]}"
            )
        self._matriz = novos if self._matriz is None else np.vstack([self._matriz, novos])
        self.trechos.extend(trechos)

    def conferir(self, embedder: Embedder) -> None:
        if embedder.nome != self.embedder_nome:
            raise IndiceIncompativel(
                f"O índice foi criado com '{self.embedder_nome}' e a busca está usando "
                f"'{embedder.nome}'. Vetores de modelos diferentes não se comparam: "
                "reindexe com o modelo novo."
            )

    def buscar(self, vetor: list[float], k: int = 4) -> list[tuple[Trecho, float]]:
        """Os k trechos mais parecidos, com a similaridade de cosseno (de -1 a 1)."""
        if self._matriz is None:
            return []
        q = np.asarray(vetor, dtype=np.float32)
        if q.shape[0] != self._matriz.shape[1]:
            raise IndiceIncompativel(
                f"pergunta com {q.shape[0]} dimensões num índice de {self._matriz.shape[1]}"
            )
        # vetores normalizados: produto interno = cosseno
        scores = self._matriz @ q
        k = min(k, len(self.trechos))
        melhores = np.argsort(-scores)[:k]
        return [(self.trechos[i], float(scores[i])) for i in melhores]

    def salvar(self, caminho: str | Path) -> None:
        dados = {
            "embedder": self.embedder_nome,
            "dimensao": self.dimensao,
            "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trechos": [asdict(t) for t in self.trechos],
            "vetores": [] if self._matriz is None else self._matriz.round(6).tolist(),
        }
        Path(caminho).write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def carregar(cls, caminho: str | Path) -> "Indice":
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        indice = cls(dados["embedder"])
        indice.adicionar([Trecho(**t) for t in dados["trechos"]], dados["vetores"])
        return indice
