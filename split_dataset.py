import os


DATA_PATH = os.path.join(os.getcwd(), 'data')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
INFO_PATH = os.path.join(DATA_PATH, 'info')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')

VARIANTS_FILE_NAME = 'variants.txt'


def create_subdirectories() -> None:
    with open(os.path.join(INFO_PATH, VARIANTS_FILE_NAME)) as infile:
        variants = infile.read().splitlines()
        for variant in variants:
            if '/' in variant:
                print(variant)
                variant = variant.replace('/', '_')

            new_dir = os.path.join(DATASET_PATH, variant)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
