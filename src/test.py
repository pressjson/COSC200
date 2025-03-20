#!/usr/bin/env python3

import network

from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

new_image = Image.open("path_to_your_image.jpg")
input_tensor = transform(new_image).unsqueeze(0)

with torch.no_grad():
    enhanced_tensor = model(input_tensor)

enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze())
enhanced_image.save("enhanced_image.jpg")


"""
@TODO: make this work. this should take in an image, chunk it, use the neural network on all the chunks, and then stitch it back together
"""
