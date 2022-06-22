import os

DATA_PATH: str = os.path.join(os.getcwd(), "data")
IMAGES_PATH: str = os.path.join(DATA_PATH, "images")
INFO_PATH: str = os.path.join(DATA_PATH, "info")
DATASET_PATH: str = os.path.join(DATA_PATH, "dataset")

VARIANTS_FILE_NAME: str = "variants.txt"
TEST_AND_VALIDATE_CLASS_SIZE: int = 15


def create_subdirectories() -> None:
    """
    Reads the file with all the possible variants and creates directories for them for 'train', 'test' and 'validate'.
    :return: None
    """
    variants = []
    with open(os.path.join(INFO_PATH, VARIANTS_FILE_NAME)) as infile:
        variants = infile.read().splitlines()

    # Get rid of the backslash in the variant name to prevent errors when creating a directory
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


def read_images_indexes(file_name: str) -> list:
    """
    Reads the given file.
    :param file_name: Name of the file to read from.
    :return: List of the image index and variant pairs,
    """
    indexes = []
    indexes_split = []
    with open(os.path.join(INFO_PATH, file_name)) as infile:
        indexes = infile.read().splitlines()

    # Get rid of the backslash in the variant name to match created directories
    for index in indexes:
        if "/" in index:
            index = index.replace("/", "_")
        indexes_split.append(index.split(maxsplit=1))

    return indexes_split


def move_images(train_entries: list, test_entries: list, validate_entries: list) -> bool:
    f"""
    For each entry type move the images associated with indexes in the entries. For 'test' and 'validation' it takes
    {TEST_AND_VALIDATE_CLASS_SIZE} images for each class and the rest is moved to 'train'.
    :param train_entries: List of the images indexes for training.
    :param test_entries: List of the images indexes for testing.
    :param validate_entries: List of the images indexes for validation.
    :return: True if all files were moved correctly, False otherwise.
    """
    for entry in train_entries:
        if not move_single_image(entry[0], entry[1], "train"):
            return False

    counter = 0
    previous_variant_name = test_entries[0][1]
    for entry in test_entries:
        if entry[1] == previous_variant_name:
            counter += 1
        else:
            counter = 1
            previous_variant_name = entry[1]

        if counter <= TEST_AND_VALIDATE_CLASS_SIZE:
            if not move_single_image(entry[0], entry[1], "test"):
                return False
        else:
            if not move_single_image(entry[0], entry[1], "train"):
                return False

    counter = 0
    previous_variant_name = validate_entries[0][1]
    for entry in validate_entries:
        if entry[1] == previous_variant_name:
            counter += 1
        else:
            counter = 1
            previous_variant_name = entry[1]

        if counter <= TEST_AND_VALIDATE_CLASS_SIZE:
            if not move_single_image(entry[0], entry[1], "validate"):
                return False
        else:
            if not move_single_image(entry[0], entry[1], "train"):
                return False

    return True


def move_single_image(index: str, variant_name: str, set_type: str) -> bool:
    """
    Moves the image to correct subdirectory.
    :param index: Index of the image to be moved.
    :param variant_name: Name of variant (subdirectory).
    :param set_type: Type of the set image should be moved to ('train', 'test' or 'validate').
    :return: True if image was moved correctly, False otherwise.
    """
    try:
        image_path = os.path.join(IMAGES_PATH, index + ".jpg")
        target_path = os.path.join(os.path.join(os.path.join(DATASET_PATH, set_type), variant_name), index + ".jpg")
        os.rename(image_path, target_path)
        return True
    except FileNotFoundError:
        # Check if the image is already moved
        if not os.path.exists(target_path):
            return False
        else:
            return True  # File is already moved


def split_dataset() -> bool:
    """
    Runs functions to create directories, read indexes and move all images.
    :return: True if everything was done correctly, False otherwise.
    """
    create_subdirectories()

    indexes_train = read_images_indexes("images_variant_train.txt")
    indexes_test = read_images_indexes("images_variant_test.txt")
    indexes_validate = read_images_indexes("images_variant_validate.txt")

    success = move_images(indexes_train, indexes_test, indexes_validate)
    return success
