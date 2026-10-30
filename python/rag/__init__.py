"""RAG do zero: um RAG pequeno, legível e completo, para aprender e adaptar."""

from .env import carregar_env

carregar_env()


from .chunking import Trecho, dividir_markdown
from .indice import Indice, IndiceIncompativel
from .pipeline import Resposta, indexar, montar_prompt, responder
from .provedores import ErroDeConfiguracao, criar_embedder, criar_gerador

__all__ = [
    "Trecho", "dividir_markdown", "Indice", "IndiceIncompativel", "Resposta", "indexar",
    "montar_prompt", "responder", "ErroDeConfiguracao", "criar_embedder", "criar_gerador",
]
