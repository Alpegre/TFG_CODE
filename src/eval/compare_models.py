"""Bloque 4 — Comparación de modelos

Lee las métricas de evaluación en validación (JSON) del perceptrón y del MLP y
genera una tabla comparativa.

Entradas:
- `results/metrics/perceptron_eval.json`
- `results/metrics/mlp_eval.json`

Salidas:
- `results/metrics/model_comparison.csv`
- `results/metrics/model_comparison.json`

Ejecución:
- `python -m src.eval.compare_models`
"""

import json
import os
import pandas as pd


def load_accuracy(path):
    """Carga la accuracy desde un JSON con formato `{"accuracy": ...}`."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["accuracy"]


def main():
    perceptron_path = "results/metrics/perceptron_eval.json"
    mlp_path = "results/metrics/mlp_eval.json"

    acc_perceptron = load_accuracy(perceptron_path)
    acc_mlp = load_accuracy(mlp_path)

    comparison = pd.DataFrame([
        {"modelo": "Perceptrón", "accuracy_val": acc_perceptron},
        {"modelo": "MLP (1 capa oculta, 64)", "accuracy_val": acc_mlp},
    ])

    os.makedirs("results/metrics", exist_ok=True)

    comparison.to_csv("results/metrics/model_comparison.csv", index=False)
    comparison.to_json("results/metrics/model_comparison.json", orient="records", indent=2)

    print("\nComparación de modelos:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()