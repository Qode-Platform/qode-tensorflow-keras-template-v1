"""tf.data input pipeline."""

import tensorflow as tf
from keras.datasets import mnist

AUTOTUNE = tf.data.AUTOTUNE


def datasets(batch_size: int = 64):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(AUTOTUNE)
    return train, test
