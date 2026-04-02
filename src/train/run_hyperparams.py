import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pat_loader import load_pat
from src.models.perceptron import build_perceptron
from src.models.mlp import build_mlp


class LossThreshold(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("loss") is not None and logs["loss"] <= self.threshold:
            self.model.stop_training = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_once(build_fn, X, y, lr, max_epochs, dmax, run_id, model_name, out_dir):
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

    return {
        "model": model_name,
        "learning_rate": lr,
        "run": run_id,
        "epochs_run": epochs_run,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "min_loss": min_loss,
        "max_accuracy": max_acc,
        "history_path": hist_path
    }


def main():
    X, y = load_pat("data/raw/letterstrain.pat")

    learning_rates = [0.2, 2.0, 5.0, 10.0]
    repeats = 5
    dmax = 0.1

    os.makedirs("results/metrics", exist_ok=True)

    results = []

    # 1) Perceptrón
    for lr in learning_rates:
        for run in range(1, repeats + 1):
            set_seed(1000 + run)
            row = train_once(
                build_fn=build_perceptron,
                X=X,
                y=y,
                lr=lr,
                max_epochs=200,
                dmax=dmax,
                run_id=run,
                model_name="perceptron",
                out_dir="results/metrics"
            )
            results.append(row)

    # 2) MLP (35 → 64 → 26)
    for lr in learning_rates:
        for run in range(1, repeats + 1):
            set_seed(2000 + run)
            row = train_once(
                build_fn=lambda learning_rate: build_mlp(hidden_units=64, learning_rate=learning_rate),
                X=X,
                y=y,
                lr=lr,
                max_epochs=300,
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
        mean_epochs=("epochs_run", "mean")
    ).reset_index()

    summary.to_csv("results/metrics/hyperparam_summary.csv", index=False)

    print("✅ Experimentos terminados. Resultados guardados en results/metrics/.")


if __name__ == "__main__":
    main()