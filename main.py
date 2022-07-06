from split_dataset import split_dataset
from cnn import build_and_train_model, make_single_prediction


def main() -> int:
    success = split_dataset()
    if not success:
        print("Error splitting dataset")
        return 0

    print("Dataset split correctly")
    cnn = build_and_train_model()

    while True:
        file_name = input('Provide file name to be predicted: ')
        make_single_prediction(file_name, cnn)


if __name__ == '__main__':
    main()

