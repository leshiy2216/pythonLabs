import argparse
import os


class DatasetIterator:
    def __init__(self, dataset_path: str, class_name: str):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.class_instances = self._get_class_instances()
        self.current_index = 0

    def _get_class_instances(self):
        instances = []
        class_folder = os.path.join(self.dataset_path, self.class_name)

        if os.path.exists(class_folder) and os.path.isdir(class_folder):
            instances = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]

        return instances

    def get_next_instance(self):
        if self.current_index < len(self.class_instances):
            next_instance = self.class_instances[self.current_index]
            self.current_index += 1
            return next_instance
        else:
            return None


class ClassIterator:
    def __init__(self, dataset_path: str, class_names: list):
        self.dataset_path = dataset_path
        self.class_names = class_names
        self.dataset_iterators = {name: DatasetIterator(dataset_path, name) for name in class_names}
        self.current_name_index = 0

    def next_image(self):
        """
        function get path to next image
        Parameters
        ----------
        """
        if not self.class_names or self.current_name_index >= len(self.class_names):
            return None

        current_class_name = self.class_names[self.current_name_index]
        current_dataset_iterator = self.dataset_iterators[current_class_name]
        next_instance = current_dataset_iterator.get_next_instance()

        if next_instance is not None:
            return next_instance
        else:
            self.current_name_index += 1
            return self.next_image()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('class_names', type=str, nargs='+', help='List of class names')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    class_names = args.class_names

    iterator = ClassIterator(dataset_path, class_names)

    instance = iterator.next_image()
    while instance is not None:
        print(instance)
        instance = iterator.next_image()