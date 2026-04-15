"""Bloque 2 — Etiquetas (A–Z)

Funciones auxiliares para convertir entre:
- índice (0–25) ↔ letra ('A'–'Z')
- vector one-hot (26) → letra

Usado por scripts de evaluación y visualización.
"""

import string

LETTERS = list(string.ascii_uppercase)


def index_to_letter(index: int) -> str:
    """Convierte un índice (0–25) en su letra correspondiente ('A'–'Z')."""
    return LETTERS[index]


def onehot_to_letter(onehot) -> str:
    """Convierte un vector one-hot (26) en una letra ('A'–'Z')."""
    idx = int(onehot.argmax())
    return index_to_letter(idx)
