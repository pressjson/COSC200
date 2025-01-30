#!/usr/bin/env python3


import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import transformers

import loader


class ImageDataset(Dataset):
    def __init__(self, X_images, Y_images, transform=None, target_transform=None):
        self.X_images = X_images
        self.Y_images = Y_images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X_images)

    def __getitem__(self, index):
        image = self.X_images[i]
        label = self.Y_images[i]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# from the pytorch tutorial


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.X_images)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# from stackoverflow
# https://stackoverflow.com/questions/65260432/how-to-create-a-neural-network-that-takes-in-an-image-and-ouputs-another-image
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    #     """ encoder """
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
    #     self.batchnorm1 = nn.BatchNorm2d(32)

    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=3)
    #     self.batchnorm2 = nn.BatchNorm2d(64)

    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=3)
    #     self.batchnorm3 = nn.BatchNorm2d(128)

    #     self.maxpool2x2 = nn.MaxPool2d(2)  # not in usage

    #     """ decoder """
    #     self.upsample2x2 = nn.Upsample(scale_factor=2)  # not in usage

    #     self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=3)
    #     self.batchnorm1 = nn.BatchNorm2d(64)

    #     self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=3)
    #     self.batchnorm2 = nn.BatchNorm2d(32)

    #     self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=(5, 5))
    #     self.batchnorm3 = nn.BatchNorm2d(3)

    # def forward(
    #     self,
    #     x,
    #     train_: bool = True,
    #     print_: bool = False,
    #     return_bottlenecks: bool = False,
    # ):
    #     """encoder"""
    #     x = self.conv1(x)
    #     x = self.batchnorm1(x)
    #     x = F.relu(x)

    #     x = self.conv2(x)
    #     x = self.batchnorm2(x)
    #     x = F.relu(x)

    #     x = self.conv3(x)
    #     x = self.batchnorm3(x)
    #     bottlenecks = F.relu(x)

    #     """ decoder """
    #     x = self.deconv1(bottlenecks)
    #     x = self.batchnorm1(x)
    #     x = F.relu(x)

    #     x = self.deconv2(x)
    #     x = self.batchnorm2(x)
    #     x = F.relu(x)

    #     x = self.deconv3(x)
    #     x = torch.sigmoid(x)

    #     return x


X_train = []
Y_train = []
for directory in os.listdir("../data"):
    PATH = "../data/"
    i = 1
    for image in os.listdir(PATH + directory + "/hq"):
        print("Loading " + PATH + directory + "/lq/png-" + str(i) + ".png")
        X_train.append(
            loader.load_file(PATH + directory + "/lq/png-" + str(i) + ".png")
        )
        print("Loading " + PATH + directory + "/hq/png-" + str(i) + ".png")
        Y_train.append(
            loader.load_file(PATH + directory + "/hq/png-" + str(i) + ".png")
        )
        i = i + 1
        break  # for testing purposes


training_dataloader = ImageDataset(X_train, Y_train)

# from the pytorch tutorial
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

model = transformers.ViTModel(training_dataloader).to(device)
print(model)


learning_rate = 1e-3
batch_size = 2

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_dataloader, model, loss_fn, optimizer)
print("Done!")

# class CustomImageDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#        self.images = images  # Input image tensor
#         self.labels = labels  # Corresponding target labels or images
#         self.transform = transform  # Optional transformations
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#
#         # Apply any transformations if needed
#         if self.transform:
#             image = self.transform(image)
#             label = self.transform(label)  # Apply transformation to target if needed
#
#         return image, label
#
#
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(2 * 48470400, 2)  # To match the flattened size
#         self.fc2 = nn.Linear(2, 2 * 48470400 * 2)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2)  # Max pooling
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)  # Max pooling
#         x = x.view(x.size(0), -1)  # Flatten for fully connected layers
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)  # Output layer
#         return x
#
#
# if True:
#     X_train = []
#     Y_train = []
#     for directory in os.listdir("../data"):
#         path = "../data/"
#         i = 1
#         for image in os.listdir(path + directory + "/hq"):
#             print("Loading " + path + directory + "/lq/png-" + str(i) + ".png")
#             X_train.append (
#                 loader.load_file(path + directory + "/lq/png-" + str(i) + ".png")
#             )
#             print("Loading " + path + directory + "/hq/png-" + str(i) + ".png")
#             Y_train.append (
#                loader.load_file(path + directory + "/hq/png-" + str(i) + ".png")
#             )
#             i = i + 1
#
#     train_dataset = CustomImageDataset(X_train, Y_train)
#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#
#     print("Initializing the neural network")
#     model = SimpleCNN()
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     print("Training the neural network")
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         print (
#             f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
#         )

#     print("Evaluating the accuracy")
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in train_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     print(f"Accuracy on the dataset: {accuracy:.2f}%")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# import numpy as np
# from PIL import Image


# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(UNet, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.middle = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.decoder = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
#         )

#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         out = self.decoder(x2)
#         return out


# # Example dataset (blurred and high-quality image pairs)
# class ImagePairDataset(Dataset):
#     def __init__(self, blurred_images, high_quality_images, transform=None):
#         self.blurred_images = blurred_images
#         self.high_quality_images = high_quality_images
#         self.transform = transform

#     def __len__(self):
#         return len(self.blurred_images)

#     def __getitem__(self, idx):
#         blurred_image = Image.open(self.blurred_images[idx])
#         high_quality_image = Image.open(self.high_quality_images[idx])

#         if self.transform:
#             blurred_image = self.transform(blurred_image)
#             high_quality_image = self.transform(high_quality_image)

#         return blurred_image, high_quality_image


# # Example data loading and training loop
# def train_model(model, dataloader, num_epochs=10):
#     criterion = nn.MSELoss()  # Mean Squared Error Loss for image reconstruction
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, targets in dataloader:
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(inputs)

#             # Compute loss
#             loss = criterion(outputs, targets)
#             loss.backward()

#             # Update parameters
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")


# # Transformations to normalize and prepare the images
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),  # Convert image to tensor (channels x height x width)
#         transforms.reshape((1024, 1024)),  # Resize for consistent input size
#     ]
# )

# # Example file paths (replace with actual paths)
# blurred_images = ["path_to_blurred_image1.jpg", "path_to_blurred_image2.jpg"]
# high_quality_images = [
#     "path_to_high_quality_image1.jpg",
#     "path_to_high_quality_image2.jpg",
# ]

# # Create dataset and dataloader
# dataset = ImagePairDataset(blurred_images, high_quality_images, transform)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Instantiate the model
# model = UNet()

# # Train the model
# train_model(model, dataloader)
