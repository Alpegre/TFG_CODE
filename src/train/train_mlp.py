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