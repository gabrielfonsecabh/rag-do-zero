import pytest

from rag.chunking import dividir_markdown

DOC = """# Manual

## Parte A

Primeiro parágrafo da parte A.

## Parte B

### Detalhe

Texto do detalhe.
"""


def test_um_trecho_por_secao_com_caminho():
    trechos = dividir_markdown(DOC, "m.md")
    assert [t.secao for t in trechos] == ["Manual > Parte A", "Manual > Parte B > Detalhe"]
    assert trechos[0].texto_para_embedding.startswith("Manual > Parte A\n")


def test_secao_grande_quebra_sem_passar_do_limite_e_com_sobreposicao():
    corpo = " ".join(f"Frase número {i} sobre o assunto." for i in range(60))
    trechos = dividir_markdown(f"# T\n\n{corpo}", "t.md", max_chars=300, sobreposicao=60)
    assert len(trechos) > 1
    assert all(len(t.texto) <= 300 + 60 for t in trechos)
    # o começo do segundo trecho repete o fim do primeiro
    assert trechos[1].texto.split()[0] in trechos[0].texto


def test_ids_unicos():
    corpo = "\n\n".join("parágrafo " * 30 for _ in range(10))
    trechos = dividir_markdown(f"# T\n\n{corpo}", "t.md", max_chars=400, sobreposicao=50)
    assert len({t.id for t in trechos}) == len(trechos)


def test_sobreposicao_maior_que_o_trecho_e_erro():
    with pytest.raises(ValueError):
        dividir_markdown(DOC, "m.md", max_chars=100, sobreposicao=100)
