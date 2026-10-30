"""Lê o arquivo .env, sem dependência externa.

Procura .env na pasta atual e na raiz do repositório. Variável já definida no
terminal vence a do arquivo.
"""

import os
from pathlib import Path


def carregar_env() -> None:
    raiz = Path(__file__).resolve().parents[2]
    for caminho in (Path.cwd() / ".env", raiz / "python" / ".env", raiz / ".env"):
        if not caminho.is_file():
            continue
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha:
                continue
            chave, valor = linha.split("=", 1)
            os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))
