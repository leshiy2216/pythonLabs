from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QHBoxLayout
from iterator import ClassIterator
from annotation import create_annotation_file
from dataset_random_copy import copy_dataset
import os


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.folder_path = ""
        self.annotation_file_path = ""
        self.dest_folder_path = ""

        self.class_iterator = None
        self.dataset_iterator = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label_folder = QLabel("Select the source dataset folder:")
        layout.addWidget(self.label_folder)

        self.btn_browse_folder = QPushButton("Browse")
        self.btn_browse_folder.clicked.connect(self.browse_folder)
        layout.addWidget(self.btn_browse_folder)

        self.label_annotation = QLabel("Enter the annotation file path:")
        layout.addWidget(self.label_annotation)

        self.edit_annotation = QLineEdit()
        layout.addWidget(self.edit_annotation)

        self.btn_create_annotation = QPushButton("Create Annotation File")
        self.btn_create_annotation.clicked.connect(self.create_annotation)
        layout.addWidget(self.btn_create_annotation)

        self.label_dest_folder = QLabel("Select the destination folder:")
        layout.addWidget(self.label_dest_folder)

        self.btn_browse_dest_folder = QPushButton("Browse")
        self.btn_browse_dest_folder.clicked.connect(self.browse_dest_folder)
        layout.addWidget(self.btn_browse_dest_folder)

        self.btn_copy_dataset = QPushButton("Copy and Rename Dataset")
        self.btn_copy_dataset.clicked.connect(self.copy_dataset)
        layout.addWidget(self.btn_copy_dataset)

        self.setLayout(layout)
        self.setWindowTitle("Dataset Processing App")

        btn_layout = QHBoxLayout()

        self.btn_next_cat = QPushButton("Next cat")
        self.btn_next_cat.clicked.connect(self.show_next_cat)
        btn_layout.addWidget(self.btn_next_cat)

        self.btn_next_dog = QPushButton("Next dog")
        self.btn_next_dog.clicked.connect(self.show_next_dog)
        btn_layout.addWidget(self.btn_next_dog)

        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setWindowTitle("App for processing dataset")

    def show_next_cat(self):
        self.show_next_instance("cat")

    def show_next_dog(self):
        self.show_next_instance("dog")

    def show_next_instance(self, target_class):
        if self.class_iterator is not None:
            next_instance = next(self.class_iterator)
            if next_instance is not None:
                print(f"Displaying the following {target_class}: {next_instance}")
            else:
                print(f"There are no more instances {target_class}.")
        else:
            print("The dataset is not loaded. Please copy the dataset first.")

    def browse_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, 'Select Source Dataset Folder')
        self.label_folder.setText(f"Selected Source Dataset Folder: {self.folder_path}")

    def create_annotation(self):
        self.annotation_file_path, _ = QFileDialog.getSaveFileName(self, 'Save Annotation File', '', 'CSV Files (*.csv)')
        self.edit_annotation.setText(self.annotation_file_path)

        create_annotation_file(self.folder_path, os.listdir(self.folder_path), self.annotation_file_path)

    def browse_dest_folder(self):
        self.dest_folder_path = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        self.label_dest_folder.setText(f"Selected Destination Folder: {self.dest_folder_path}")

    def copy_dataset(self):
        copy_dataset(self.folder_path, self.dest_folder_path, randomize=False)

        self.class_iterator = ClassIterator(self.annotation_file_path, os.listdir(self.folder_path))
        self.dataset_iterator = iter(self.class_iterator)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())