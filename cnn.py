import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from split_dataset import DATASET_PATH

BATCH_SIZE: int = 32
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
IMG_DEPTH: int = 3
EPOCHS: int = 40

NUMBER_OF_CLASSES: int = 100
class_names: list = []


def build_and_train_model() -> tf.keras.Sequential:
    """
    Creates train and validation datasets, creates the model object and trains it.
    :return: Trained model of the CNN.
    """
    global class_names

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        shuffle=True,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, 'validate'),
        shuffle=True,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_dataset.class_names

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

    # Build model
    cnn = build_EfficientNetB0()
    cnn.summary()

    # Train model
    results = cnn.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

    # Visualize the training results
    visualize_training_results(results)

    return cnn


def build_cnn() -> tf.keras.Sequential:
    """
    Creates the CNN architecture.
    :return: Compiled model of the CNN.
    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)))
    model.add(data_augmentation())

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=NUMBER_OF_CLASSES))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model


def build_EfficientNetB0() -> tf.keras.Sequential:
    """
    Creates the EfficientNetB0 model with data augmentation.
    :return: Compiled model of the CNN.
    """
    input_layers = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    data_augmentation_layer = data_augmentation()
    output_layers = tf.keras.applications.EfficientNetB0(
        include_top=True, weights=None, classes=NUMBER_OF_CLASSES)(data_augmentation_layer(input_layers))

    model = tf.keras.Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def build_AlexNet() -> tf.keras.Sequential:
    """
    Creates the AlexNet model.
    :return: Compiled model of the CNN.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu',
                                     input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), kernel_initializer='he_normal'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu',
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

    model.add(tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     kernel_initializer='he_normal'))

    model.add(tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     kernel_initializer='he_normal'))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     kernel_initializer='he_normal'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def data_augmentation() -> tf.keras.Sequential:
    """
    Creates a data augmentation layer.
    :return: Data augmentation layer.
    """
    data_augmentation_layer = tf.keras.Sequential()
    data_augmentation_layer.add(tf.keras.layers.RandomFlip('horizontal',
                                                           input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)))
    data_augmentation_layer.add(tf.keras.layers.RandomRotation(0.1))
    data_augmentation_layer.add(tf.keras.layers.RandomZoom(0.1))

    return data_augmentation_layer


def visualize_training_results(results: tf.keras.callbacks.History) -> None:
    """
    Visualize the accuracy and the loss of the training history.
    :param results: History of the training.
    :return: None
    """
    accuracy = results.history['accuracy']
    validation_accuracy = results.history['val_accuracy']

    loss = results.history['loss']
    validation_loss = results.history['val_loss']

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


def make_single_prediction(image_path: str, cnn: tf.keras.Sequential) -> None:
    """
    Predict the type of the airplane based on the given image.
    :param image_path: Path to the image to be predicted.
    :param cnn: Model of the trained CNN.
    :return: None
    """
    test_image = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = tf.keras.utils.img_to_array(test_image)
    image_array = tf.expand_dims(image_array, 0)  # Create a batch

    predictions = cnn.predict(image_array)
    score = tf.nn.softmax(predictions[0])

    print(f'This airplane is most likely {class_names[np.argmax(score)]}, '
          f'with a {round(100 * np.max(score))}% confidence.')

