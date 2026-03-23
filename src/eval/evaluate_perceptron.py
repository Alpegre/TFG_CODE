import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from src.data.pat_loader import load_pat


def main():
    X_val, y_val = load_pat("data/raw/lettersval.pat")

    model = tf.keras.models.load_model("results/logs/perceptron_model.keras")
    y_pred = model.predict(X_val)

    y_true_labels = np.argmax(y_val, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Guardar métricas en JSON
    with open("results/metrics/perceptron_eval.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc}, f, indent=4)

    # Guardar matriz de confusión como figura
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de confusión - Perceptrón")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("results/figures/perceptron_confusion.png")
    plt.close()

    print("Accuracy validación:", acc)


if __name__ == "__main__":
    main()