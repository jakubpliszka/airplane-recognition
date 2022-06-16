import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    cnn = tf.keras.Sequential()

    project_path = os.getcwd()
    dataset_path = os.path.join(os.path.join(project_path, 'data'), 'images')

