import pytest

from rag import Indice, IndiceIncompativel, criar_gerador, indexar, responder
from rag.avaliacao import avaliar_busca, carregar_perguntas
from rag.provedores import EmbedderBrinquedo


@pytest.fixture(scope="module")
def embedder():
    return EmbedderBrinquedo()


@pytest.fixture(scope="module")
def indice(embedder):
    return indexar(embedder)


def test_indexa_o_corpus_inteiro(indice):
    assert len(indice) > 10
    assert {t.arquivo for t in indice.trechos} >= {"ferias.md", "reembolso.md"}


def test_busca_acha_a_fonte_certa(indice, embedder):
    achados = indice.buscar(embedder.embed_pergunta("Quantos dias de férias posso vender?"), k=3)
    assert achados[0][0].arquivo == "ferias.md"
    # contando palavras, "férias" e "dias" pesam mais que "vender": o trecho do
    # abono vem entre os 3 primeiros, não necessariamente em 1º (ver lição 01)
    assert any("10 dias" in t.texto for t, _ in achados)


def test_resultados_em_ordem_de_similaridade(indice, embedder):
    scores = [s for _, s in indice.buscar(embedder.embed_pergunta("reembolso de viagem"), k=5)]
    assert scores == sorted(scores, reverse=True)


def test_prompt_numera_trechos_e_cita_fonte(indice, embedder):
    r = responder("Qual o limite de alimentação?", indice, embedder, criar_gerador("nenhum"), k=3)
    assert r.texto is None
    assert "[1] (fonte: reembolso.md" in r.prompt
    assert r.prompt.rstrip().endswith("Pergunta: Qual o limite de alimentação?")


def test_similaridade_minima_corta_o_irrelevante(indice, embedder):
    r = responder("xyzzy plugh", indice, embedder, criar_gerador("nenhum"), similaridade_minima=0.2)
    assert r.trechos == []
    assert "(nenhum trecho encontrado)" in r.prompt


def test_salvar_e_carregar(tmp_path, indice, embedder):
    arquivo = tmp_path / "indice.json"
    indice.salvar(arquivo)
    copia = Indice.carregar(arquivo)
    q = embedder.embed_pergunta("senha mínima")
    assert [t.id for t, _ in copia.buscar(q, 3)] == [t.id for t, _ in indice.buscar(q, 3)]


def test_recusa_buscar_com_modelo_diferente(indice):
    outro = EmbedderBrinquedo(dimensao=128)
    with pytest.raises(IndiceIncompativel, match="reindexe"):
        responder("férias", indice, outro, criar_gerador("nenhum"))


def test_avaliacao_da_linha_de_base(indice, embedder):
    r = avaliar_busca(indice, embedder, carregar_perguntas(), k=3)
    # pergunta sem fonte (resposta certa: "não sei") fica fora da conta
    assert len(r.resultados) == 16
    assert r.acerto >= 0.8 and 0 < r.mrr <= 1
