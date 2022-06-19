import os

from split_dataset import split_dataset
from cnn import create_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main() -> int:
    success = split_dataset()
    if not success:
        print("Error splitting dataset")
        return 0

    print("Dataset split correctly")
    create_dataset()


if __name__ == '__main__':
    main()

