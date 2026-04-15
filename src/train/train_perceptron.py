"""Bloque 3 — Entrenamiento del perceptrón

Script de entrenamiento del perceptrón (35 → 26) con softmax.

- Carga `data/raw/letterstrain.pat`.
- Entrena el modelo.
- Guarda el modelo y el histórico para su análisis posterior.

Salidas:
- `results/logs/perceptron_model.keras`
- `results/metrics/perceptron_history.csv`

Ejecución:
- `python -m src.train.train_perceptron`
"""

import os
import pandas as pd

from src.data.pat_loader import load_pat
from src.models.perceptron import build_perceptron


def main():
    X, y = load_pat("data/raw/letterstrain.pat")

    model = build_perceptron(learning_rate=0.1)
    history = model.fit(X, y, epochs=200, verbose=1)

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    model.save("results/logs/perceptron_model.keras")

    # Guardar history en CSV
    df = pd.DataFrame(history.history)
    df.to_csv("results/metrics/perceptron_history.csv", index=False)


if __name__ == "__main__":
    main()