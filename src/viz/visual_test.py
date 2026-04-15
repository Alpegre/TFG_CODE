"""Bloque 2 — Test visual de carga

Carga un patrón del dataset de entrenamiento y lo muestra en 7×5 para verificar
que el parsing del `.pat` es correcto.

Uso:
    python -m src.viz.visual_test
"""

from src.data.pat_loader import load_pat
from src.data.labels import onehot_to_letter
from src.viz.plot_letter import plot_letter


def main():
    X, y = load_pat("data/raw/letterstrain.pat")

    idx = 10
    letter = onehot_to_letter(y[idx])
    plot_letter(X[idx], title=f"Letra: {letter}")


if __name__ == "__main__":
    main()
