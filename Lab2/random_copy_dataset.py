import os
import shutil
import random
import argparse

def copy_and_randomize_dataset(src_folder: str, dest_folder: str) -> None:
    """
    function create dataset copy and give them random numbers
    Parameters
    ----------
    src_folder: str
    dest_folder: str
    """
    try:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for class_folder in os.listdir(src_folder):
            class_path = os.path.join(src_folder, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    src_filepath = os.path.join(class_path, filename)
                    random_number = random.randint(0, 10000)
                    dest_filename = f"{random_number}.jpg"
                    dest_filepath = os.path.join(dest_folder, dest_filename)

                    shutil.copy(src_filepath, dest_filepath)

        print("dataset copied and randomized")
    except Exception as e:
        print(f"error copying and randomizing dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy and randomize dataset')
    parser.add_argument('src_folder', type=str, help='Source dataset directory')
    parser.add_argument('dest_folder', type=str, help='Destination directory for copied dataset')

    args = parser.parse_args()

    copy_and_randomize_dataset(args.src_folder, args.dest_folder)