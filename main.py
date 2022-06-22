from split_dataset import split_dataset
from cnn import build_cnn


def main() -> int:
    success = split_dataset()
    if not success:
        print("Error splitting dataset")
        return 0

    print("Dataset split correctly")
    build_cnn()


if __name__ == '__main__':
    main()

