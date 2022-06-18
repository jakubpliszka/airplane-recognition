import os


DATA_PATH = os.path.join(os.getcwd(), "data")
IMAGES_PATH = os.path.join(DATA_PATH, "images")
INFO_PATH = os.path.join(DATA_PATH, "info")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")

VARIANTS_FILE_NAME = "variants.txt"


def create_subdirectories() -> None:
    variants = []
    with open(os.path.join(INFO_PATH, VARIANTS_FILE_NAME)) as infile:
        variants = infile.read().splitlines()

    for variant in variants:
        if "/" in variant:
            variant = variant.replace("/", "_")

        train_dir = os.path.join(DATASET_PATH, "train")
        if not os.path.exists(os.path.join(train_dir, variant)):
            os.mkdir(os.path.join(train_dir, variant))

        test_dir = os.path.join(DATASET_PATH, "test")
        if not os.path.exists(os.path.join(test_dir, variant)):
            os.mkdir(os.path.join(test_dir, variant))

        validate_dir = os.path.join(DATASET_PATH, "validate")
        if not os.path.exists(os.path.join(validate_dir, variant)):
            os.mkdir(os.path.join(validate_dir, variant))


def read_images_indexes(file_name : str) -> list:
    indexes = []
    indexes_split = []
    with open(os.path.join(INFO_PATH, file_name)) as infile:
        indexes = infile.read().splitlines()

    for index in indexes:
        if "/" in index:
            index = index.replace("/", "_")
        indexes_split.append(index.split(maxsplit=1))

    return indexes_split


def move_images(indexes : list, set_type : str) -> bool:
    for index in indexes:
        try:
            image_path = os.path.join(IMAGES_PATH, index[0] + ".jpg")
            target_path = os.path.join(os.path.join(os.path.join(DATASET_PATH, set_type), index[1]), index[0] + ".jpg")
            os.rename(image_path, target_path)
        except FileNotFoundError:
            # Check if the image is already moved
            if not os.path.exists(target_path):
                return False
            # else do nothing, file is already moved

    return True


def split_dataset():
    create_subdirectories()
    indexes_train = read_images_indexes("images_variant_train.txt")
    indexes_test = read_images_indexes("images_variant_test.txt")
    indexes_validate = read_images_indexes("images_variant_validate.txt")
    move_images(indexes_train, "train")
    move_images(indexes_test, "test")
    move_images(indexes_validate, "validate")
