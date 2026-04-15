"""Bloque 4 — Entrenamiento del MLP

Script de entrenamiento de un MLP sencillo (35 → hidden → 26) con softmax.

- Carga `data/raw/letterstrain.pat`.
- Entrena el modelo.
- Guarda el modelo y el histórico para su análisis posterior.

Salidas:
- `results/logs/mlp_model.keras`
- `results/metrics/mlp_history.csv`

Ejecución:
- `python -m src.train.train_mlp`
"""

import os
import pandas as pd

from src.data.pat_loader import load_pat
from src.models.mlp import build_mlp


def main():
    X, y = load_pat("data/raw/letterstrain.pat")

    model = build_mlp(hidden_units=64, learning_rate=0.01)
    history = model.fit(X, y, epochs=300, verbose=1)

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    model.save("results/logs/mlp_model.keras")

    # Guardar history en CSV
    df = pd.DataFrame(history.history)
    df.to_csv("results/metrics/mlp_history.csv", index=False)


if __name__ == "__main__":
    main()