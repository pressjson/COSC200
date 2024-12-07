#!/usr/bin/env python3

import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def load_file(path: str):
    image = Image.open(path).convert("RGB")
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor
