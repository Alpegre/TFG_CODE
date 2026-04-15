"""Bloque 4 — MLP (una capa oculta)

Define un MLP: 35 → hidden_units (ReLU) → 26 (softmax).
Se entrena con Adam y entropía cruzada categórica.

Usado por:
- src/train/train_mlp.py
- src/train/run_hyperparams.py
"""

import tensorflow as tf


def build_mlp(input_dim=35, hidden_units=128, output_dim=26, learning_rate=0.001):
    """Crea y compila un MLP con una capa oculta."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(output_dim, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
