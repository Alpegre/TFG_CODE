"""Bloque 5 — Visualización del barrido de hiperparámetros

Genera gráficas a partir de `results/metrics/hyperparam_summary.csv` para
analizar el efecto del *learning rate* y la estabilidad entre repeticiones.

Salidas (en `results/figures/`):
- `hyperparam_loss_vs_lr.png`
- `hyperparam_acc_vs_lr.png`
- `hyperparam_stability_loss.png`

Ejecución:
- `python -m src.viz.plot_hyperparams`
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    summary_path = os.path.join(METRICS_DIR, "hyperparam_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("No existe results/metrics/hyperparam_summary.csv. Ejecuta run_hyperparams primero.")

    df = pd.read_csv(summary_path)

    # --- Gráfico 1: loss final medio vs LR ---
    plt.figure(figsize=(7, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["learning_rate"], sub["mean_final_loss"], marker="o", label=model)
    plt.xscale("log")
    plt.xlabel("Learning rate (log)")
    plt.ylabel("Loss final media")
    plt.title("Efecto del learning rate en la loss final")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hyperparam_loss_vs_lr.png"), dpi=300)
    plt.close()

    # --- Gráfico 2: accuracy final media vs LR ---
    plt.figure(figsize=(7, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["learning_rate"], sub["mean_final_acc"], marker="o", label=model)
    plt.xscale("log")
    plt.xlabel("Learning rate (log)")
    plt.ylabel("Accuracy final media")
    plt.title("Efecto del learning rate en la accuracy final")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hyperparam_acc_vs_lr.png"), dpi=300)
    plt.close()

    # --- Gráfico 3: estabilidad (desviación típica) ---
    plt.figure(figsize=(7, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["learning_rate"], sub["std_final_loss"], marker="o", label=model)
    plt.xscale("log")
    plt.xlabel("Learning rate (log)")
    plt.ylabel("Std de loss final")
    plt.title("Estabilidad por learning rate (desviación típica)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hyperparam_stability_loss.png"), dpi=300)
    plt.close()

    print("✅ Gráficas de hiperparámetros generadas en results/figures/")

if __name__ == "__main__":
    main()