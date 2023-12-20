import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import Image
import glob
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any

def load_data(csv_path: str) -> List[str]:
    """
    func load data from csv
    Parameters
    ----------
    csv_path : str
    """
    dframe = pd.read_csv(csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"])
    return dframe["Absolute path"].tolist()

def split_data(images: List[str], test_size: float = 0.1, val_size: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """
    func split data to train, test and valid
    Parameters
    ----------
    images : List[str]
    test_size : float, optional
    val_size : float, optional
    """
    train_data, test_val_data = train_test_split(images, test_size=(test_size + val_size), random_state=42)
    test_data, valid_data = train_test_split(test_val_data, test_size=(val_size / (test_size + val_size)), random_state=42)
    return train_data, test_data, valid_data

class CustomDataset(Dataset):
    def __init__(self, data: List[str], labels: List[int], transform: Any = None):
        """
        create custom dataset
        Parameters
        ----------
        data : List[str]
        labels : List[int]
        transform : Any, optional
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        method for get img and him label
        Parameters
        ----------
        index : int
        """
        img_path = self.data[index]
        img = Image.open(img_path)
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        label = self.labels[index]
        return img, label

def transform_data(train_list: List[str], test_list: List[str], valid_list: List[str]) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
    """
    transform data

    Parameters
    ----------
    train_list : List[str]
    test_list : List[str]
    valid_list : List[str]
    """
    custom_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_labels = [0 if "cat" in path else 1 for path in train_list]
    test_labels = [0 if "cat" in path else 1 for path in test_list]
    valid_labels = [0 if "cat" in path else 1 for path in valid_list]

    train_data = CustomDataset(train_list, train_labels, transform=custom_transforms)
    test_data = CustomDataset(test_list, test_labels, transform=custom_transforms)
    valid_data = CustomDataset(valid_list, valid_labels, transform=custom_transforms)

    return train_data, test_data, valid_data