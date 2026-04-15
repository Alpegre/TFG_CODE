"""Bloque 2 — Visualización de patrones 7×5

`plot_letter()` convierte un vector de 35 valores binarios en una matriz 7×5 y lo visualiza.

Uso (desde otros scripts):
    from src.viz.plot_letter import plot_letter
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_letter(pattern, title=None):
    """Recibe un vector de 35 valores (0/1) y lo muestra como matriz 7×5."""
    if len(pattern) != 35:
        raise ValueError("El patrón debe tener exactamente 35 valores.")

    matrix = np.array(pattern).reshape(7, 5)

    plt.imshow(matrix, cmap="Greys", interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    if title:
        plt.title(title)
    plt.show()
