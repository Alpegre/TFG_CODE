import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"

def ensure_dirs():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def write_table(df: pd.DataFrame, base_name: str, caption: str, label: str) -> None:
    """Exporta una tabla en CSV/LaTeX/Markdown dentro de results/metrics."""
    csv_path = os.path.join(METRICS_DIR, f"{base_name}.csv")
    tex_path = os.path.join(METRICS_DIR, f"{base_name}.tex")
    md_path = os.path.join(METRICS_DIR, f"{base_name}.md")

    df.to_csv(csv_path, index=False)

    latex_table = df.to_latex(index=False, caption=caption, label=label)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    md_table = df.to_markdown(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

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
    write_table(
        comp,
        base_name="table_model_comparison",
        caption="Comparación de accuracy en validación",
        label="tab:comparacion_modelos",
    )


def generate_hyperparam_table_for_memory() -> None:
    path = os.path.join(METRICS_DIR, "hyperparam_summary.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    preferred = [
        "model",
        "learning_rate",
        "mean_final_loss",
        "std_final_loss",
        "mean_final_acc",
        "std_final_acc",
        "mean_val_loss",
        "std_val_loss",
        "mean_val_acc",
        "std_val_acc",
        "mean_epochs",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df = df.sort_values(["model", "learning_rate"], kind="stable")
    write_table(
        df,
        base_name="table_hyperparam_summary",
        caption="Resumen de hiperparámetros (media y desviación típica)",
        label="tab:hyperparam_summary",
    )


def generate_noise_table_for_memory() -> None:
    path = os.path.join(METRICS_DIR, "noise_eval_summary.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    preferred = ["model", "noise_pixels", "repeats", "mean_accuracy", "std_accuracy"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    df = df.sort_values(["model", "noise_pixels"], kind="stable")

    write_table(
        df,
        base_name="table_noise_eval_summary",
        caption="Robustez con ruido (media y desviación típica)",
        label="tab:noise_eval_summary",
    )

def main():
    ensure_dirs()
    plot_loss_and_accuracy()
    plot_bar_comparison()
    generate_table_for_memory()
    generate_hyperparam_table_for_memory()
    generate_noise_table_for_memory()
    print("✅ Tablas y gráficas generadas en results/figures y results/metrics.")

if __name__ == "__main__":
    main()