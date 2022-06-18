import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import os

from split_dataset import create_subdirectories

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    create_subdirectories()
    print("ok")

