import argparse
import logging
import os
import random
import shutil


def copy_dataset(src_folder: str, dest_folder: str, randomize: bool = False) -> None:
    """
    function do copy and renamed or randomized dataset.
    Parameters
    ----------
    src_folder: str
    dest_folder: str
    randomize: bool
    """
    try:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for class_folder in os.listdir(src_folder):
            class_path = os.path.join(src_folder, class_folder)
            if os.path.isdir(class_path):
                for idx, filename in enumerate(os.listdir(class_path)):
                    src_filepath = os.path.join(class_path, filename)

                    if randomize:
                        random_number = random.randint(0, 10000)
                        dest_filename = f"{random_number}.jpg"
                    else:
                        class_name = f"{class_folder}_"
                        dest_filename = f"{class_name}{idx:04}.jpg"

                    dest_filepath = os.path.join(dest_folder, dest_filename)

                    shutil.copy(src_filepath, dest_filepath)

        logging.info(f"Dataset copied and {'randomized' if randomize else 'renamed'}")
    except Exception as e:
        logging.error(f"error copying and {'randomizing' if randomize else 'renaming'} dataset: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy and rename or randomimize dataset.')
    parser.add_argument('src_folder', type=str, help='Source folder path')
    parser.add_argument('dest_folder', type=str, help='Destination folder path')
    parser.add_argument('--randomize', action='store_true', help='Do you need to assign random numbers?')

    args = parser.parse_args()

    copy_dataset(args.src_folder, args.dest_folder, randomize=args.randomize)