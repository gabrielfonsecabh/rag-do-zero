"""Deixa as lições importarem o pacote `rag` sem instalar nada."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def titulo(texto: str) -> None:
    print(f"\n{texto}\n{'=' * len(texto)}")
