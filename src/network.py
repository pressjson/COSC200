#!/usr/bin/env python3

import loader

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # Input image tensor
        self.labels = labels  # Corresponding target labels or images
        self.transform = transform  # Optional transformations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply any transformations if needed
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)  # Apply transformation to target if needed

        return image, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1
        )  # 3 input channels (RGB), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Flattened size after conv layers
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Max pooling
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # Max pooling
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x


if True:
    X_train = []
    Y_train = []
    for directory in os.listdir("../data"):
        path = "../data/"
        i = 1
        for image in os.listdir(path + directory + "/hq"):
            print("Loading " + path + directory + "/lq/png-" + str(i) + ".png")
            X_train.append(
                loader.load_file(path + directory + "/lq/png-" + str(i) + ".png")
            )
            print("Loading " + path + directory + "/hq/png-" + str(i) + ".png")
            Y_train.append(
                loader.load_file(path + directory + "/hq/png-" + str(i) + ".png")
            )
            i = i + 1

    train_dataset = CustomImageDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Initializing the neural network")
    model = SimpleCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training the neural network")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )

    print("Evaluating the accuracy")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the dataset: {accuracy:.2f}%")
