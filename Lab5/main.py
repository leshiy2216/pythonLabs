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
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any


def load_data(cat_csv_path : str, dog_csv_path : str) -> Tuple[List[str], List[str]]:
    """
    func load data from csv
    Parameters
    ----------
    cat_csv_path : str
    dog_csv_path : str
    """
    cat_dframe = pd.read_csv(cat_csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"])
    dog_dframe = pd.read_csv(dog_csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"])

    cat_images = cat_dframe["Absolute path"].tolist()
    dog_images = dog_dframe["Absolute path"].tolist()

    return cat_images, dog_images


def split_data(cat_images : List[str], dog_images : List[str], test_size : float = 0.1, val_size : float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """
    func split data to train, test and valid
    Parameters
    ----------
    cat_images : List[str]
    dog_images : List[str]
    test_size : float, optional
    val_size : float, optional
    """
    cat_train_data, cat_test_val_data = train_test_split(cat_images, test_size=(test_size + val_size), random_state=42)
    cat_test_data, cat_valid_data = train_test_split(cat_test_val_data, test_size=(val_size / (test_size + val_size)), random_state=42)

    dog_train_data, dog_test_val_data = train_test_split(dog_images, test_size=(test_size + val_size), random_state=42)
    dog_test_data, dog_valid_data = train_test_split(dog_test_val_data, test_size=(val_size / (test_size + val_size)), random_state=42)

    train_data = cat_train_data + dog_train_data
    test_data = cat_test_data + dog_test_data
    valid_data = cat_valid_data + dog_valid_data

    return train_data, test_data, valid_data


class CustomDataset(Dataset):
    def __init__(self, data : List[str], labels : List[int], transform : Any = None):
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
        """
        get the length of dataset
        """
        return len(self.data)

    def __getitem__(self, index : int) -> Tuple[torch.Tensor, int]:
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


def transform_data(train_list : List[str], test_list : List[str], valid_list : List[str]) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
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


class Cnn(nn.Module):
    """
    Convolutional neural network model
    """

    def __init__(self) -> None:
        super(Cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor):
        """
        Forward pass of the CNN model.
        Parameters
        ----------
        x : torch.Tensor
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
def show_results(epochs : int, acc : List[float], loss : List[float], v_acc : List[float], v_loss : List[float]) -> None:
    """
    creates graphs based on the learning results for train and validation sets.
    Parameters
    ----------
    epochs : int
    acc : List[float]
    loss : List[float]
    v_acc : List[float]
    v_loss : List[float]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(range(epochs), acc, color="orange", label="Train accuracy")
    ax2.plot(range(epochs), loss, color="orange", label="Train loss")
    
    ax1.plot(range(epochs), v_acc, color="steelblue", label="Validation accuracy")
    ax2.plot(range(epochs), v_loss, color="steelblue", label="Validation loss")
    
    ax1.legend()
    ax2.legend()
    plt.show()

def train_loop(epochs: int, batch_size: int, lear: float, train_data: CustomDataset, test_data: CustomDataset, valid_data: CustomDataset) -> Tuple[list, Cnn]:
    """
    create, train model
    Parameters
    ----------
    epochs: Number of epochs.
    batch_size: List of training accuracies.
    loss: List of training losses.
    v_acc: List of validation accuracies.
    v_loss: List of validation losses.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)
    
    model = Cnn()
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lear)
    criterion = nn.CrossEntropyLoss()

    accuracy_values = []
    loss_values = []
    valid_accuracy_values = []
    valid_loss_values = []

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=batch_size, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(f"Epoch : {epoch + 1}, train accuracy : {epoch_accuracy}, train loss : {epoch_loss}")
        accuracy_values.append(epoch_accuracy.item())
        loss_values.append(epoch_loss.item())

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0

            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

            print(f"Epoch : {epoch + 1}, val_accuracy : {epoch_val_accuracy}, val_loss : {epoch_val_loss}")
            valid_accuracy_values.append(epoch_val_accuracy.item())
            valid_loss_values.append(epoch_val_loss.item())

        model.train()

    show_results(epochs, accuracy_values, loss_values, valid_accuracy_values, valid_loss_values)

    rose_probs = []
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=100, shuffle=False
    )
    with torch.no_grad():
        for data, fileid in test_loader:
            data = data.to(device)
            preds = model(data)
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
            rose_probs += list(zip(list(fileid), preds_list))

    rose_probs.sort(key=lambda x: int(x[0]))
    
    return rose_probs, model

if __name__ == "__main__":
    cat_images, dog_images = load_data('C:\\Users\\User\\Desktop\\testing\\dataset\\cat_annotation.csv', 'C:\\Users\\User\\Desktop\\testing\\dataset\\dog_annotation.csv')
    
    trenka, testik, valid = split_data(cat_images, dog_images)

    print(len(trenka))
    print(len(testik))
    print(len(valid))