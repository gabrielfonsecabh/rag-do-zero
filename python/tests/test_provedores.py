import math

import pytest

from rag.provedores import (
    EmbedderBrinquedo,
    ErroDeConfiguracao,
    GeradorNenhum,
    criar_embedder,
    criar_gerador,
)


def test_padrao_de_geracao_e_nenhum(monkeypatch):
    monkeypatch.delenv("GERACAO_PROVEDOR", raising=False)
    assert isinstance(criar_gerador(), GeradorNenhum)


def test_troca_por_variavel_de_ambiente(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVEDOR", "brinquedo")
    assert isinstance(criar_embedder(), EmbedderBrinquedo)


def test_argumento_vence_a_variavel(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVEDOR", "voyage")
    assert isinstance(criar_embedder("brinquedo"), EmbedderBrinquedo)


def test_provedor_inexistente_diz_as_opcoes():
    with pytest.raises(ErroDeConfiguracao, match="Opções: brinquedo, local, openai, voyage"):
        criar_embedder("pinecone")


@pytest.mark.parametrize(
    "tipo,provedor,variavel",
    [
        ("embedder", "openai", "OPENAI_API_KEY"),
        ("embedder", "voyage", "VOYAGE_API_KEY"),
        ("gerador", "anthropic", "ANTHROPIC_API_KEY"),
        ("gerador", "openai", "OPENAI_API_KEY"),
    ],
)
def test_provedor_pago_sem_chave_explica_o_que_falta(monkeypatch, tipo, provedor, variavel):
    monkeypatch.delenv(variavel, raising=False)
    criar = criar_embedder if tipo == "embedder" else criar_gerador
    with pytest.raises(ErroDeConfiguracao, match=variavel):
        criar(provedor)


def test_provedor_pago_usa_modelo_da_variavel(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "teste")
    monkeypatch.setenv("EMBEDDINGS_MODELO", "text-embedding-3-large")
    assert criar_embedder("openai").nome == "openai:text-embedding-3-large"


def test_openai_monta_a_chamada_certa(monkeypatch):
    chamadas = []

    def falso(url, cabecalhos, corpo, tempo=60):
        chamadas.append((url, cabecalhos, corpo))
        return {"data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(corpo["input"]))]}

    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setattr("rag.provedores._post_json", falso)
    vetores = criar_embedder("openai").embed_documentos(["a", "b"])
    url, cab, corpo = chamadas[0]
    assert url.endswith("/embeddings") and cab["Authorization"] == "Bearer sk-teste"
    assert corpo == {"model": "text-embedding-3-small", "input": ["a", "b"]}
    assert vetores == [[0.6, 0.8], [0.6, 0.8]]  # normalizado


def test_voyage_diferencia_documento_de_pergunta(monkeypatch):
    tipos = []

    def falso(url, cabecalhos, corpo, tempo=60):
        tipos.append(corpo["input_type"])
        return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setenv("VOYAGE_API_KEY", "v")
    monkeypatch.setattr("rag.provedores._post_json", falso)
    e = criar_embedder("voyage")
    e.embed_documentos(["x"])
    e.embed_pergunta("y")
    assert tipos == ["document", "query"]


def test_anthropic_monta_a_chamada_certa(monkeypatch):
    visto = {}

    def falso(url, cabecalhos, corpo, tempo=60):
        visto.update(url=url, cab=cabecalhos, corpo=corpo)
        return {"content": [{"type": "text", "text": "ok [1]"}]}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setattr("rag.provedores._post_json", falso)
    assert criar_gerador("anthropic").gerar("sistema", "pergunta") == "ok [1]"
    assert visto["url"] == "https://api.anthropic.com/v1/messages"
    assert visto["cab"]["anthropic-version"] == "2023-06-01"
    assert visto["corpo"]["system"] == "sistema"


def test_brinquedo_e_deterministico_e_normalizado():
    e = EmbedderBrinquedo()
    a, b = e.embed_pergunta("Férias de verão"), e.embed_pergunta("Férias de verão")
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)


def test_brinquedo_ignora_acento_e_maiuscula():
    e = EmbedderBrinquedo()
    assert e.embed_pergunta("FÉRIAS") == e.embed_pergunta("ferias")
