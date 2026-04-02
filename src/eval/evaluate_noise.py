import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf

from src.data.labels import index_to_letter
from src.data.pat_loader import load_pat


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


def flip_pixels(X: np.ndarray, n_flips: int, rng: np.random.Generator) -> np.ndarray:
    """Flip (0↔1) exactly n_flips pixels per sample.

    Assumes binary inputs in {0,1}. For n_flips=0 returns a copy.
    """
    if n_flips < 0:
        raise ValueError("n_flips must be >= 0")

    X_noisy = np.array(X, copy=True)
    if n_flips == 0:
        return X_noisy

    if X_noisy.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X_noisy.shape}")

    n_samples, n_features = X_noisy.shape
    if n_flips > n_features:
        raise ValueError(f"n_flips ({n_flips}) cannot be > n_features ({n_features})")

    for i in range(n_samples):
        idx = rng.choice(n_features, size=n_flips, replace=False)
        X_noisy[i, idx] = 1.0 - X_noisy[i, idx]

    return X_noisy


def evaluate_model_with_noise(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_true_labels: np.ndarray,
    noise_pixels: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Returns: (runs_df, cm_sum, per_sample_df)."""
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    labels = np.unique(y_true_labels)
    labels_sorted = np.sort(labels)

    cm_sum = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=np.int64)
    per_sample_mis = np.zeros(len(y_true_labels), dtype=np.int64)

    rows: list[dict] = []

    for run in range(1, repeats + 1):
        rng = np.random.default_rng(seed + run)
        X_noisy = flip_pixels(X_val, noise_pixels, rng)

        y_pred = model.predict(X_noisy, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)

        acc = float(accuracy_score(y_true_labels, y_pred_labels))
        rows.append({
            "noise_pixels": noise_pixels,
            "run": run,
            "seed": int(seed + run),
            "accuracy": acc,
        })

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels_sorted)
        cm_sum += cm

        per_sample_mis += (y_pred_labels != y_true_labels).astype(np.int64)

    runs_df = pd.DataFrame(rows)

    per_sample_df = pd.DataFrame({
        "noise_pixels": noise_pixels,
        "sample": np.arange(1, len(y_true_labels) + 1),
        "true_label": y_true_labels,
        "true_letter": [index_to_letter(int(i)) for i in y_true_labels],
        "misclass_rate": per_sample_mis / float(repeats),
    })

    return runs_df, cm_sum, per_sample_df


def save_confusion_matrix(cm: np.ndarray, labels: list[int], out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[index_to_letter(int(i)) for i in labels],
        yticklabels=[index_to_letter(int(i)) for i in labels],
    )
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluación con ruido (flip de píxeles) sobre lettersval.pat",
    )
    p.add_argument(
        "--noise",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Nº de píxeles a invertir por patrón (por defecto: 2 4 6)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Repeticiones por nivel de ruido (por defecto: 100)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Semilla base para reproducibilidad (por defecto: 12345)",
    )
    p.add_argument(
        "--val-path",
        type=str,
        default="data/raw/lettersval.pat",
        help="Ruta del fichero de validación .pat",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_val, y_val = load_pat(args.val_path)
    y_true_labels = np.argmax(y_val, axis=1)
    labels_sorted = np.sort(np.unique(y_true_labels))

    models = [
        ModelSpec(name="perceptron", path="results/logs/perceptron_model.keras"),
        ModelSpec(name="mlp", path="results/logs/mlp_model.keras"),
    ]

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_runs: list[pd.DataFrame] = []
    all_summary: list[dict] = []
    all_per_sample: list[pd.DataFrame] = []

    for spec in models:
        if not os.path.exists(spec.path):
            raise FileNotFoundError(
                f"No se encuentra el modelo '{spec.name}' en {spec.path}. Ejecuta el entrenamiento antes."
            )

        model = tf.keras.models.load_model(spec.path)

        for n in args.noise:
            runs_df, cm_sum, per_sample_df = evaluate_model_with_noise(
                model=model,
                X_val=X_val,
                y_true_labels=y_true_labels,
                noise_pixels=int(n),
                repeats=int(args.repeats),
                seed=int(args.seed),
            )

            runs_df.insert(0, "model", spec.name)
            per_sample_df.insert(0, "model", spec.name)

            all_runs.append(runs_df)
            all_per_sample.append(per_sample_df)

            mean_acc = float(runs_df["accuracy"].mean())
            std_acc = float(runs_df["accuracy"].std(ddof=1)) if len(runs_df) > 1 else 0.0

            all_summary.append({
                "model": spec.name,
                "noise_pixels": int(n),
                "repeats": int(args.repeats),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
            })

            cm_path = f"results/figures/{spec.name}_confusion_noise{int(n)}.png"
            save_confusion_matrix(
                cm=cm_sum,
                labels=list(map(int, labels_sorted)),
                out_path=cm_path,
                title=f"Matriz de confusión - {spec.name} - ruido {int(n)} px",
            )

    runs_out = pd.concat(all_runs, ignore_index=True)
    summary_out = pd.DataFrame(all_summary).sort_values(["model", "noise_pixels"]).reset_index(drop=True)
    per_sample_out = pd.concat(all_per_sample, ignore_index=True)

    runs_out.to_csv("results/metrics/noise_eval_runs.csv", index=False)
    summary_out.to_csv("results/metrics/noise_eval_summary.csv", index=False)
    per_sample_out.to_csv("results/metrics/noise_eval_per_sample.csv", index=False)

    # Figura: accuracy vs ruido
    plt.figure(figsize=(7, 4))
    for model_name in summary_out["model"].unique():
        sub = summary_out[summary_out["model"] == model_name]
        plt.errorbar(
            sub["noise_pixels"],
            sub["mean_accuracy"],
            yerr=sub["std_accuracy"],
            marker="o",
            capsize=3,
            label=model_name,
        )

    plt.xlabel("Píxeles invertidos (ruido)")
    plt.ylabel("Accuracy (validación)")
    plt.title("Robustez al ruido (media ± std)")
    plt.ylim(0, 1.05)
    plt.xticks(sorted(set(int(x) for x in summary_out["noise_pixels"].tolist())))
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/noise_accuracy_vs_pixels.png", dpi=300)
    plt.close()

    # Un JSON compacto por si lo quieres cargar fácil en la memoria
    with open("results/metrics/noise_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out.to_dict(orient="records"), f, indent=2)

    print("✅ Evaluación con ruido terminada.")
    print(summary_out.to_string(index=False))


if __name__ == "__main__":
    main()
