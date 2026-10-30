"""O RAG de ponta a ponta: indexar, buscar, montar o prompt, gerar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Trecho, dividir_markdown
from .indice import Indice
from .provedores import Embedder, Gerador

PASTA_CORPUS = Path(__file__).resolve().parents[2] / "corpus"

SISTEMA = """Você responde perguntas usando SOMENTE os trechos de documentos fornecidos.

Regras:
- Se a resposta não estiver nos trechos, diga que não encontrou essa informação nos documentos. Não complete com conhecimento próprio.
- Cite a fonte de cada afirmação com o número do trecho entre colchetes, por exemplo [2].
- Responda em português, de forma direta, em poucas frases."""


@dataclass
class Resposta:
    pergunta: str
    trechos: list[tuple[Trecho, float]]
    prompt: str
    texto: str | None  # None quando GERACAO_PROVEDOR=nenhum


def carregar_documentos(pasta: str | Path = PASTA_CORPUS) -> list[tuple[str, str]]:
    pasta = Path(pasta)
    arquivos = sorted(pasta.glob("**/*.md")) + sorted(pasta.glob("**/*.txt"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum .md ou .txt em {pasta}")
    return [(str(a.relative_to(pasta)), a.read_text(encoding="utf-8")) for a in arquivos]


def indexar(
    embedder: Embedder,
    pasta: str | Path = PASTA_CORPUS,
    max_chars: int = 800,
    sobreposicao: int = 150,
) -> Indice:
    trechos: list[Trecho] = []
    for nome, texto in carregar_documentos(pasta):
        trechos.extend(dividir_markdown(texto, nome, max_chars, sobreposicao))
    indice = Indice(embedder.nome)
    vetores = embedder.embed_documentos([t.texto_para_embedding for t in trechos])
    indice.adicionar(trechos, vetores)
    return indice


def montar_prompt(pergunta: str, trechos: list[tuple[Trecho, float]]) -> str:
    blocos = [
        f"[{i}] (fonte: {t.arquivo}, seção: {t.secao})\n{t.texto}"
        for i, (t, _score) in enumerate(trechos, start=1)
    ]
    contexto = "\n\n".join(blocos) if blocos else "(nenhum trecho encontrado)"
    return f"Trechos dos documentos:\n\n{contexto}\n\nPergunta: {pergunta}"


def responder(
    pergunta: str,
    indice: Indice,
    embedder: Embedder,
    gerador: Gerador,
    k: int = 4,
    similaridade_minima: float | None = None,
) -> Resposta:
    indice.conferir(embedder)
    achados = indice.buscar(embedder.embed_pergunta(pergunta), k=k)
    if similaridade_minima is not None:
        achados = [(t, s) for t, s in achados if s >= similaridade_minima]
    prompt = montar_prompt(pergunta, achados)
    return Resposta(pergunta, achados, prompt, gerador.gerar(SISTEMA, prompt))
