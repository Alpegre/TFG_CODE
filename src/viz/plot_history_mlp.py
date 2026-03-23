import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/metrics/mlp_history.csv")

    os.makedirs("results/figures", exist_ok=True)

    # Curva de loss
    plt.figure()
    plt.plot(df["loss"], label="loss")
    plt.title("Loss por época - MLP")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/mlp_loss.png")
    plt.close()

    # Curva de accuracy
    plt.figure()
    plt.plot(df["accuracy"], label="accuracy")
    plt.title("Accuracy por época - MLP")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/mlp_accuracy.png")
    plt.close()


if __name__ == "__main__":
    main()