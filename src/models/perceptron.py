import tensorflow as tf


def build_perceptron(input_dim=35, output_dim=26, learning_rate=0.1):
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