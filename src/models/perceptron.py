"""Bloque 3 — Perceptrón (modelo base)

Define un perceptrón multiclasificación: 35 entradas → 26 clases, con salida softmax.
Se entrena con SGD y pérdida de entropía cruzada categórica.

Usado por:
- src/train/train_perceptron.py
- src/train/run_hyperparams.py
"""

import tensorflow as tf


def build_perceptron(input_dim=35, output_dim=26, learning_rate=0.1):
    """Crea y compila un perceptrón (una capa densa softmax)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(output_dim, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
