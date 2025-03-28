#!/usr/bin/env python3


import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# thanks perplexity (;


class ImageDataset(Dataset):
    def __init__(self, root_dir="../data/chunks", transform=None):
        self.lq_images = []
        self.hq_images = []
        self.transform = transform
        self.root_dir = root_dir

        for subdirectory in os.listdir(root_dir):
            for filename in os.listdir(os.path.join(root_dir, subdirectory)):
                if "hq" in filename:
                    self.hq_images.append(
                        os.path.join(root_dir, subdirectory, filename)
                    )
                else:
                    self.lq_images.append(
                        os.path.join(root_dir, subdirectory, filename)
                    )

    def __len__(self):
        return len(self.hq_images)

    def __getitem__(self, i):
        lq_image = Image.open(self.lq_images[i]).convert("RGB")
        hq_image = Image.open(self.hq_images[i]).convert("RGB")
        if self.transform:
            lq_image = self.transform(lq_image)
            hq_image = self.transform(hq_image)
        return lq_image, hq_image


class ImageEnhancementNet(nn.Module):
    def __init__(self):
        super(ImageEnhancementNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def make_model(num_epochs=10):

    print("Training for {num_epochs} epochs")

    # Set up dataset and dataloader

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageDataset(root_dir="../data/chunks", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEnhancementNet().to(device)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Training loop
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        model.train()
        for lq_batch, hq_batch in dataloader:
            lq_batch, hq_batch = lq_batch.to(device), hq_batch.to(device)

            optimizer.zero_grad()
            outputs = model(lq_batch)
            loss = criterion(outputs, hq_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluation and visualization
    model.eval()
    with torch.no_grad():
        # Get a sample image from the dataset
        lq_sample, hq_sample = dataset[0]
        lq_sample = lq_sample.unsqueeze(0).to(device)

        # Generate enhanced image
        enhanced_sample = model(lq_sample)

        # Convert tensors to images for visualization
        lq_img = transforms.ToPILImage()(lq_sample.squeeze().cpu())
        hq_img = transforms.ToPILImage()(hq_sample)
        enhanced_img = transforms.ToPILImage()(enhanced_sample.squeeze().cpu())

    # Save the model
    torch.save(model.state_dict(), "{num_epochs}_epochs_model.pth".format(num_epochs=num_epochs))


if __name__ == "__main__":
    model_ranges = [1, 5, 10, 30, 50, 100, 200, 500]
    for i in model_ranges:
        make_model(num_epochs = i)
