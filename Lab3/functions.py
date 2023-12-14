import argparse
import csv
import logging
import os
import random
import shutil


def create_annotation_file(folder_path: str, subfolder_paths: list, annotation_file_path: str):
    try:
        with open(annotation_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['The absolute path', 'relative path', 'the text name of the class'])

            for subfolder_path in subfolder_paths:
                class_name = os.path.basename(subfolder_path)

                for filename in os.listdir(os.path.join(folder_path, subfolder_path)):
                    absolute_path = os.path.join(folder_path, subfolder_path, filename)
                    relative_path = os.path.join(subfolder_path, filename)
                    csv_writer.writerow([absolute_path, relative_path, class_name])

        logging.info(f"The file with the annotation has been created: {annotation_file_path}")
    except Exception as e:
        logging.exception(f"Error in creating an annotation file: {e}")


def copy_dataset(src_folder: str, dest_folder: str, randomize: bool = False) -> None:
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


class DatasetIterator:
    def __init__(self, annotation_file: str, class_name: str):
        self.annotation_file = annotation_file
        self.class_name = class_name
        self.class_instances = None
        self.current_index = 0

    def __iter__(self):
        self.class_instances = self._get_class_instances()
        return self

    def _get_class_instances(self):
        instances = []
        with open(self.annotation_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) == 3 and row[2] == self.class_name:
                    instances.append(row[0])
        return instances

    def __next__(self):
        if self.current_index < len(self.class_instances):
            next_instance = self.class_instances[self.current_index]
            self.current_index += 1
            return next_instance
        else:
            raise StopIteration


class ClassIterator:
    def __init__(self, annotation_file: str, class_names: list):
        self.annotation_file = annotation_file
        self.class_names = class_names
        self.current_name_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_name_index < len(self.class_names):
            current_class_name = self.class_names[self.current_name_index]
            self.current_name_index += 1
            return DatasetIterator(self.annotation_file, current_class_name)
        else:
            raise StopIteration