"""Provedores de embeddings e de geração de texto.

Todo o resto do projeto fala só com duas interfaces, `Embedder` e `Gerador`.
Para trocar de modelo, local ou pago, não se mexe em código: muda-se uma
variável de ambiente e chama-se `criar_embedder()` ou `criar_gerador()`.

    EMBEDDINGS_PROVEDOR = local | openai | voyage | brinquedo
    EMBEDDINGS_MODELO   = (opcional) nome do modelo do provedor
    GERACAO_PROVEDOR    = nenhum | anthropic | openai
    GERACAO_MODELO      = (opcional) nome do modelo do provedor

Para acrescentar um provedor novo, escreva uma classe com os mesmos métodos e
registre em `_EMBEDDERS` ou `_GERADORES`. Nada mais muda.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from typing import Callable, Protocol, Sequence

Vetor = list[float]


class ErroDeConfiguracao(RuntimeError):
    """Falta chave, pacote ou o provedor pedido não existe."""


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Transforma texto em vetor. Todo vetor sai normalizado (norma 1)."""

    nome: str  # "provedor:modelo", gravado no índice para detectar troca de modelo

    def embed_documentos(self, textos: Sequence[str]) -> list[Vetor]: ...

    def embed_pergunta(self, texto: str) -> Vetor: ...


class Gerador(Protocol):
    """Gera a resposta final a partir do prompt montado com o contexto."""

    nome: str

    def gerar(self, sistema: str, mensagem: str) -> str | None: ...


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def normalizar(v: Sequence[float]) -> Vetor:
    norma = math.sqrt(sum(x * x for x in v))
    return [x / norma for x in v] if norma > 0 else list(v)


def _post_json(url: str, cabecalhos: dict[str, str], corpo: dict, tempo: int = 60) -> dict:
    """POST com JSON, sem dependência externa, com erro legível."""
    req = urllib.request.Request(
        url,
        data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json", **cabecalhos},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=tempo) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detalhe = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{url} respondeu {e.code}: {detalhe}") from e


def _chave(nome_variavel: str, provedor: str) -> str:
    valor = os.environ.get(nome_variavel, "").strip()
    if not valor:
        raise ErroDeConfiguracao(
            f"O provedor '{provedor}' precisa da variável {nome_variavel}. "
            f"Defina no .env ou no terminal: export {nome_variavel}=..."
        )
    return valor


def _em_lotes(itens: Sequence[str], tamanho: int):
    for i in range(0, len(itens), tamanho):
        yield list(itens[i : i + tamanho])


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

_PALAVRAS_VAZIAS = set(
    "a o as os de da do das dos e é em no na nos nas um uma uns umas por para com sem que "
    "se ao à às ou mais menos já até como qual quais quando onde quem meu minha".split()
)


def palavras(texto: str) -> list[str]:
    """Minúsculas, sem acento, sem pontuação e sem palavras vazias."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [p for p in re.findall(r"[a-z0-9]+", t) if p not in _PALAVRAS_VAZIAS and len(p) > 1]


class EmbedderBrinquedo:
    """Embedding "de brinquedo": conta palavras num vetor de tamanho fixo.

    Não entende sinônimo nem contexto ("notebook" e "computador" são palavras
    diferentes para ele). Serve para aprender o conceito (lição 01), para rodar
    os testes sem baixar modelo e como linha de base na avaliação.
    """

    def __init__(self, dimensao: int = 512):
        self.dimensao = dimensao
        self.nome = f"brinquedo:bag-of-words-{dimensao}"

    def _vetor(self, texto: str) -> Vetor:
        v = [0.0] * self.dimensao
        for p in palavras(texto):
            # radical simples: "ferias" e "feria" caem no mesmo lugar
            raiz = p[:6]
            i = int(hashlib.md5(raiz.encode()).hexdigest(), 16) % self.dimensao
            v[i] += 1.0
        return normalizar(v)

    def embed_documentos(self, textos: Sequence[str]) -> list[Vetor]:
        return [self._vetor(t) for t in textos]

    def embed_pergunta(self, texto: str) -> Vetor:
        return self._vetor(texto)


class EmbedderLocal:
    """Modelo que roda no seu computador, grátis, via sentence-transformers.

    O padrão é o multilingual-e5-small: 384 dimensões, bom em português e
    leve o suficiente para CPU. A família e5 espera os prefixos "query: " e
    "passage: ", e esquecer deles derruba a qualidade da busca sem dar erro.
    """

    PADRAO = "intfloat/multilingual-e5-small"

    def __init__(self, modelo: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ErroDeConfiguracao(
                "O provedor 'local' precisa do pacote sentence-transformers: "
                "pip install sentence-transformers"
            ) from e
        self.modelo = modelo or self.PADRAO
        self._st = SentenceTransformer(self.modelo)
        self.nome = f"local:{self.modelo}"
        self._e5 = "e5" in self.modelo.lower()

    def _encode(self, textos: Sequence[str]) -> list[Vetor]:
        vs = self._st.encode(list(textos), normalize_embeddings=True, batch_size=32)
        return [list(map(float, v)) for v in vs]

    def embed_documentos(self, textos: Sequence[str]) -> list[Vetor]:
        if self._e5:
            textos = [f"passage: {t}" for t in textos]
        return self._encode(textos)

    def embed_pergunta(self, texto: str) -> Vetor:
        return self._encode([f"query: {texto}" if self._e5 else texto])[0]


class EmbedderOpenAI:
    """Embeddings da OpenAI (pago). Precisa de OPENAI_API_KEY.

    OPENAI_BASE_URL permite apontar para qualquer serviço compatível.
    """

    PADRAO = "text-embedding-3-small"

    def __init__(self, modelo: str | None = None):
        self.chave = _chave("OPENAI_API_KEY", "openai")
        self.modelo = modelo or self.PADRAO
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.nome = f"openai:{self.modelo}"

    def _embed(self, textos: Sequence[str]) -> list[Vetor]:
        saida: list[Vetor] = []
        for lote in _em_lotes(textos, 100):
            r = _post_json(
                f"{self.url}/embeddings",
                {"Authorization": f"Bearer {self.chave}"},
                {"model": self.modelo, "input": lote},
            )
            ordenados = sorted(r["data"], key=lambda d: d["index"])
            saida.extend(normalizar(d["embedding"]) for d in ordenados)
        return saida

    def embed_documentos(self, textos: Sequence[str]) -> list[Vetor]:
        return self._embed(textos)

    def embed_pergunta(self, texto: str) -> Vetor:
        return self._embed([texto])[0]


class EmbedderVoyage:
    """Embeddings da Voyage AI (pago). Precisa de VOYAGE_API_KEY.

    A Voyage diferencia documento de pergunta pelo campo input_type, o mesmo
    papel dos prefixos do e5.
    """

    PADRAO = "voyage-3.5-lite"

    def __init__(self, modelo: str | None = None):
        self.chave = _chave("VOYAGE_API_KEY", "voyage")
        self.modelo = modelo or self.PADRAO
        self.nome = f"voyage:{self.modelo}"

    def _embed(self, textos: Sequence[str], tipo: str) -> list[Vetor]:
        saida: list[Vetor] = []
        for lote in _em_lotes(textos, 100):
            r = _post_json(
                "https://api.voyageai.com/v1/embeddings",
                {"Authorization": f"Bearer {self.chave}"},
                {"model": self.modelo, "input": lote, "input_type": tipo},
            )
            ordenados = sorted(r["data"], key=lambda d: d["index"])
            saida.extend(normalizar(d["embedding"]) for d in ordenados)
        return saida

    def embed_documentos(self, textos: Sequence[str]) -> list[Vetor]:
        return self._embed(textos, "document")

    def embed_pergunta(self, texto: str) -> Vetor:
        return self._embed([texto], "query")[0]


_EMBEDDERS: dict[str, Callable[[str | None], Embedder]] = {
    "brinquedo": lambda modelo: EmbedderBrinquedo(),
    "local": EmbedderLocal,
    "openai": EmbedderOpenAI,
    "voyage": EmbedderVoyage,
}


def criar_embedder(provedor: str | None = None, modelo: str | None = None) -> Embedder:
    """A função de troca. Sem argumentos, lê EMBEDDINGS_PROVEDOR e EMBEDDINGS_MODELO.

    Exemplos:
        criar_embedder()                       # local, grátis (padrão)
        criar_embedder("openai")               # precisa de OPENAI_API_KEY
        criar_embedder("voyage", "voyage-3.5") # precisa de VOYAGE_API_KEY
    """
    provedor = (provedor or os.environ.get("EMBEDDINGS_PROVEDOR") or "local").strip().lower()
    modelo = modelo or os.environ.get("EMBEDDINGS_MODELO") or None
    if provedor not in _EMBEDDERS:
        raise ErroDeConfiguracao(
            f"Provedor de embeddings '{provedor}' não existe. Opções: {', '.join(_EMBEDDERS)}"
        )
    return _EMBEDDERS[provedor](modelo)


# ---------------------------------------------------------------------------
# Geradores
# ---------------------------------------------------------------------------


class GeradorNenhum:
    """Não chama modelo nenhum. O pipeline mostra o prompt montado.

    É o padrão: dá para aprender e depurar a busca sem gastar nada, e ver
    exatamente o que um modelo receberia.
    """

    nome = "nenhum"

    def __init__(self, modelo: str | None = None):
        pass

    def gerar(self, sistema: str, mensagem: str) -> str | None:
        return None


class GeradorAnthropic:
    """Claude, da Anthropic (pago). Precisa de ANTHROPIC_API_KEY."""

    PADRAO = "claude-haiku-5-5"

    def __init__(self, modelo: str | None = None):
        self.chave = _chave("ANTHROPIC_API_KEY", "anthropic")
        self.modelo = modelo or self.PADRAO
        self.nome = f"anthropic:{self.modelo}"

    def gerar(self, sistema: str, mensagem: str) -> str | None:
        r = _post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.chave, "anthropic-version": "2023-06-01"},
            {
                "model": self.modelo,
                "max_tokens": 1024,
                "system": sistema,
                "messages": [{"role": "user", "content": mensagem}],
            },
        )
        return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")


class GeradorOpenAI:
    """Qualquer API no formato da OpenAI (chat/completions).

    Com OPENAI_BASE_URL serve também para modelos locais e grátis, como o
    Ollama (OPENAI_BASE_URL=http://localhost:11434/v1, OPENAI_API_KEY=ollama).
    """

    PADRAO = "gpt-4o-mini"

    def __init__(self, modelo: str | None = None):
        self.chave = _chave("OPENAI_API_KEY", "openai")
        self.modelo = modelo or self.PADRAO
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.nome = f"openai:{self.modelo}"

    def gerar(self, sistema: str, mensagem: str) -> str | None:
        r = _post_json(
            f"{self.url}/chat/completions",
            {"Authorization": f"Bearer {self.chave}"},
            {
                "model": self.modelo,
                "messages": [
                    {"role": "system", "content": sistema},
                    {"role": "user", "content": mensagem},
                ],
            },
        )
        return r["choices"][0]["message"]["content"]


_GERADORES: dict[str, Callable[[str | None], Gerador]] = {
    "nenhum": GeradorNenhum,
    "anthropic": GeradorAnthropic,
    "openai": GeradorOpenAI,
}


def criar_gerador(provedor: str | None = None, modelo: str | None = None) -> Gerador:
    """A função de troca da geração. Sem argumentos, lê GERACAO_PROVEDOR e GERACAO_MODELO."""
    provedor = (provedor or os.environ.get("GERACAO_PROVEDOR") or "nenhum").strip().lower()
    modelo = modelo or os.environ.get("GERACAO_MODELO") or None
    if provedor not in _GERADORES:
        raise ErroDeConfiguracao(
            f"Provedor de geração '{provedor}' não existe. Opções: {', '.join(_GERADORES)}"
        )
    return _GERADORES[provedor](modelo)
