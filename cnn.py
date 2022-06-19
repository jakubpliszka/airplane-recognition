import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from split_dataset import DATASET_PATH

BATCH_SIZE: int = 32
IMG_HEIGHT: int = 180
IMG_WIDTH: int = 180
EPOCHS: int = 10


def create_dataset() -> None:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "validate"),
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_dataset.map(lambda x, y: (normalization_layer(x), y))

    num_classes = 100

    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())

    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(num_classes))

    cnn.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    cnn.summary()

    cnn.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

