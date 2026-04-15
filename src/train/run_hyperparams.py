"""Bloque 5 — Barrido de hiperparámetros (learning rate)

Ejecuta un barrido de *learning rates* para:
- Perceptrón (Bloque 3)
- MLP (Bloque 4)

Para cada learning rate se realizan varias repeticiones (semillas distintas) y
se aplica parada temprana cuando `loss <= dmax`. Además se calcula el rendimiento
en validación (`lettersval.pat`) para poder comparar generalización.

Salidas (en `results/metrics/`):
- `hyperparam_results.csv`: resultados por ejecución.
- `hyperparam_summary.csv`: medias y desviaciones por (modelo, lr).
- `*_history.csv`: histórico por ejecución.

Ejecución:
- `python -m src.train.run_hyperparams`
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pat_loader import load_pat
from src.models.perceptron import build_perceptron
from src.models.mlp import build_mlp


class LossThreshold(tf.keras.callbacks.Callback):
    """Detiene el entrenamiento cuando la loss cae por debajo de un umbral."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("loss") is not None and logs["loss"] <= self.threshold:
            self.model.stop_training = True


def set_seed(seed: int):
    """Fija semillas de `random`, `numpy` y `tensorflow` para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_once(
    build_fn,
    X,
    y,
    X_val,
    y_val,
    lr,
    max_epochs,
    dmax,
    run_id,
    model_name,
    out_dir,
):
    """Entrena una ejecución, guarda el histórico y devuelve métricas clave."""
    model = build_fn(learning_rate=lr)

    callbacks = [
        LossThreshold(dmax),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    history = model.fit(
        X, y,
        epochs=max_epochs,
        verbose=0,
        callbacks=callbacks
    )

    hist_df = pd.DataFrame(history.history)
    hist_df["epoch"] = np.arange(1, len(hist_df) + 1)

    run_tag = f"{model_name}_lr{lr}_run{run_id}"
    hist_path = os.path.join(out_dir, f"{run_tag}_history.csv")
    hist_df.to_csv(hist_path, index=False)

    epochs_run = len(hist_df)
    final_loss = float(hist_df["loss"].iloc[-1])
    final_acc = float(hist_df["accuracy"].iloc[-1]) if "accuracy" in hist_df else np.nan
    min_loss = float(hist_df["loss"].min())
    max_acc = float(hist_df["accuracy"].max()) if "accuracy" in hist_df else np.nan

    val_loss = np.nan
    val_acc = np.nan
    if X_val is not None and y_val is not None:
        eval_out = model.evaluate(X_val, y_val, verbose=0)
        if isinstance(eval_out, (list, tuple)) and len(eval_out) >= 2:
            val_loss = float(eval_out[0])
            val_acc = float(eval_out[1])
        else:
            val_loss = float(eval_out)

    return {
        "model": model_name,
        "learning_rate": lr,
        "run": run_id,
        "epochs_run": epochs_run,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "min_loss": min_loss,
        "max_accuracy": max_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "history_path": hist_path
    }


def parse_args() -> argparse.Namespace:
    """Parsea parámetros del barrido (LRs, repeticiones, dmax, semillas, etc.)."""
    p = argparse.ArgumentParser(
        description="Bloque 5 — Barrido de hiperparámetros (learning rate) con repeticiones y dmax.",
    )
    p.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.2, 2.0, 5.0, 10.0],
        help="Lista de learning rates a probar.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeticiones por learning rate.",
    )
    p.add_argument(
        "--dmax",
        type=float,
        default=0.1,
        help="Parada temprana si loss <= dmax.",
    )
    p.add_argument(
        "--max-epochs-perceptron",
        type=int,
        default=200,
        help="Máximo de épocas para el perceptrón.",
    )
    p.add_argument(
        "--max-epochs-mlp",
        type=int,
        default=300,
        help="Máximo de épocas para el MLP.",
    )
    p.add_argument(
        "--seed-perceptron",
        type=int,
        default=1000,
        help="Semilla base para el perceptrón (se suma el nº de repetición).",
    )
    p.add_argument(
        "--seed-mlp",
        type=int,
        default=2000,
        help="Semilla base para el MLP (se suma el nº de repetición).",
    )
    p.add_argument(
        "--val-path",
        type=str,
        default="data/raw/lettersval.pat",
        help="Ruta del fichero de validación .pat para calcular val_accuracy.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    X, y = load_pat("data/raw/letterstrain.pat")
    X_val, y_val = load_pat(str(args.val_path))

    learning_rates = [float(lr) for lr in args.learning_rates]
    repeats = int(args.repeats)
    dmax = float(args.dmax)

    os.makedirs("results/metrics", exist_ok=True)

    results = []

    # 1) Perceptrón
    for lr in learning_rates:
        for run in range(1, repeats + 1):
            set_seed(int(args.seed_perceptron) + run)
            row = train_once(
                build_fn=build_perceptron,
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                lr=lr,
                max_epochs=int(args.max_epochs_perceptron),
                dmax=dmax,
                run_id=run,
                model_name="perceptron",
                out_dir="results/metrics"
            )
            results.append(row)

    # 2) MLP (35 → 64 → 26)
    for lr in learning_rates:
        for run in range(1, repeats + 1):
            set_seed(int(args.seed_mlp) + run)
            row = train_once(
                build_fn=lambda learning_rate: build_mlp(hidden_units=64, learning_rate=learning_rate),
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                lr=lr,
                max_epochs=int(args.max_epochs_mlp),
                dmax=dmax,
                run_id=run,
                model_name="mlp",
                out_dir="results/metrics"
            )
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("results/metrics/hyperparam_results.csv", index=False)

    # Resumen de estabilidad (desviación típica)
    summary = df.groupby(["model", "learning_rate"]).agg(
        mean_final_loss=("final_loss", "mean"),
        std_final_loss=("final_loss", "std"),
        mean_final_acc=("final_accuracy", "mean"),
        std_final_acc=("final_accuracy", "std"),
        mean_val_loss=("val_loss", "mean"),
        std_val_loss=("val_loss", "std"),
        mean_val_acc=("val_accuracy", "mean"),
        std_val_acc=("val_accuracy", "std"),
        mean_epochs=("epochs_run", "mean")
    ).reset_index()

    summary.to_csv("results/metrics/hyperparam_summary.csv", index=False)

    print("✅ Experimentos terminados. Resultados guardados en results/metrics/.")


if __name__ == "__main__":
    main()