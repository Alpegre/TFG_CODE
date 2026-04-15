"""Bloque 2 — Test rápido del loader

Imprime shapes y un ejemplo de patrón para verificar la carga de `letterstrain.pat`.

Uso:
    python -m src.data.test_loader
"""

from src.data.pat_loader import load_pat


def main():
    X, y = load_pat("data/raw/letterstrain.pat")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Primer patrón (entrada):", X[0])
    print("Primer patrón (salida):", y[0])


if __name__ == "__main__":
    main()
