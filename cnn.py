import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from split_dataset import DATASET_PATH

BATCH_SIZE: int = 32
IMG_HEIGHT: int = 128
IMG_WIDTH: int = 128
IMG_DEPTH: int = 3
EPOCHS: int = 40

class_names: list = []


def build_cnn() -> tf.keras.Sequential:
    """
    Creates dataset, builds and trains the model, visualizes the results of the training.
    :return: None
    """
    global class_names

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, 'validate'),
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_dataset.class_names
    number_of_classes = len(class_names)

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

    data_augmentation_layer = tf.keras.Sequential()
    data_augmentation_layer.add(tf.keras.layers.RandomFlip('horizontal',
                                                           input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)))
    data_augmentation_layer.add(tf.keras.layers.RandomRotation(0.1))
    data_augmentation_layer.add(tf.keras.layers.RandomZoom(0.1))

    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)))
    cnn.add(data_augmentation_layer)

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(2))

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(2))

    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(2))

    cnn.add(tf.keras.layers.Dropout(0.4))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=number_of_classes))

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

    return cnn


def make_single_prediction(image_path: str, cnn: tf.keras.Sequential) -> None:
    test_image = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = tf.keras.utils.img_to_array(test_image)
    image_array = tf.expand_dims(image_array, 0)  # Create a batch

    predictions = cnn.predict(image_array)
    score = tf.nn.softmax(predictions[0])

    print(f'This airplane is most likely {class_names[np.argmax(score)]}, '
          f'with a {round(100 * np.max(score))}% confidence.')
