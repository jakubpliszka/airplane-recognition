import os
import matplotlib.pyplot as plt
import tensorflow as tf

from split_dataset import DATASET_PATH


BATCH_SIZE: int = 32
IMG_HEIGHT: int = 128
IMG_WIDTH: int = 128
EPOCHS: int = 10


def build_cnn() -> None:
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

    classes_number = len(train_dataset.class_names)
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

    # Data augmentation for overfitting (did not work)
    # data_augmentation = tf.keras.Sequential()
    # data_augmentation.add(tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    # data_augmentation.add(tf.keras.layers.RandomRotation(0.1))
    # data_augmentation.add(tf.keras.layers.RandomZoom(0.1))

    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    # cnn.add(data_augmentation) (did not work)

    cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())

    cnn.add(tf.keras.layers.Dropout(0.4))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(classes_number))

    cnn.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    cnn.summary()

    result = cnn.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

    # Visualize the training results
    accuracy = result.history['accuracy']
    validation_accuracy = result.history['val_accuracy']

    loss = result.history['loss']
    validation_loss = result.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

