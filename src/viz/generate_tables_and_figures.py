import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"

def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_loss_and_accuracy():
    # Cargar históricos
    perc = pd.read_csv(os.path.join(METRICS_DIR, "perceptron_history.csv"))
    mlp = pd.read_csv(os.path.join(METRICS_DIR, "mlp_history.csv"))

    perc["epoch"] = range(1, len(perc) + 1)
    mlp["epoch"] = range(1, len(mlp) + 1)

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(perc["epoch"], perc["loss"], label="Perceptrón")
    plt.plot(mlp["epoch"], mlp["loss"], label="MLP")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "loss_comparison.png"), dpi=300)
    plt.close()

    # --- Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(perc["epoch"], perc["accuracy"], label="Perceptrón")
    plt.plot(mlp["epoch"], mlp["accuracy"], label="MLP")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Curva de precisión")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=300)
    plt.close()

def plot_bar_comparison():
    comp = pd.read_csv(os.path.join(METRICS_DIR, "model_comparison.csv"))

    plt.figure(figsize=(6, 4))
    plt.bar(comp["modelo"], comp["accuracy_val"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy (validación)")
    plt.title("Comparación de modelos")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_bar.png"), dpi=300)
    plt.close()

def generate_table_for_memory():
    comp = pd.read_csv(os.path.join(METRICS_DIR, "model_comparison.csv"))

    # Tabla en CSV limpia
    comp.to_csv(os.path.join(METRICS_DIR, "table_model_comparison.csv"), index=False)

    # Tabla en LaTeX
    latex_table = comp.to_latex(index=False, caption="Comparación de accuracy en validación", label="tab:comparacion_modelos")
    with open(os.path.join(METRICS_DIR, "table_model_comparison.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table)

    # Tabla en Markdown
    md_table = comp.to_markdown(index=False)
    with open(os.path.join(METRICS_DIR, "table_model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md_table)

def main():
    ensure_dirs()
    plot_loss_and_accuracy()
    plot_bar_comparison()
    generate_table_for_memory()
    print("✅ Tablas y gráficas generadas en results/figures y results/metrics.")

if __name__ == "__main__":
    main()