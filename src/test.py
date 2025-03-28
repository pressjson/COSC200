#!/usr/bin/env python3

import network, stitcher, imslp_bootstrap as bootstrap, chunker

from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
import os
import re


def upscale_file(input_file_directory, input_file_name, model_name="image_enhancement_model.pth"):

    # setting everything up

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = network.ImageEnhancementNet()
    model = nn.DataParallel(model)
    model.load_state_dict(
        torch.load(model_name, map_location=device)
    )
    model = model.to(device)
    model.eval()
    print(model)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # making the images out of the pdf

    bootstrap.make_images(
        os.path.join(input_file_directory, input_file_name), "../data/test/images"
    )

    # make the chunks to pass to the model

    page = 0
    if not os.path.exists("../data/test/chunks"):
        os.makedirs("../data/test/chunks")
    print("making chunks")
    for image in os.listdir("../data/test/images"):
        chunker.make_chunks(
            quality="lq",
            input_file_location=os.path.join("../data/test/images", image),
            output_dir=os.path.join("../data/test/chunks", str(page)),
        )
        page = page + 1

    # use the model to convert

    for subdirectory in os.listdir("../data/test/chunks"):
        if "upscaled" in subdirectory:
            continue

        print("Upscaling page " + subdirectory)

        path = os.path.join("../data/test/chunks", subdirectory)
        for image in os.listdir(path):

            pattern = r"(\w+)_(\d+)_(\d+)\.jpg"
            match = re.match(pattern, image)
            name, x_str, y_str = match.groups()
            if not os.path.exists(
                os.path.join("../data/test/chunks/upscaled", subdirectory)
            ):
                os.makedirs(os.path.join("../data/test/chunks/upscaled", subdirectory))

            new_image = Image.open(os.path.join(path, image)).convert("RGB")
            input_tensor = transform(new_image).unsqueeze(0).to(device)
            # print("Input tensor shape: ")
            # print(input_tensor.shape)
            # print("Input tensor: ")
            # print(input_tensor)

            with torch.no_grad():
                output = model(input_tensor)

            output_image = output.squeeze().cpu()
            output_image = output_image * 255

            # print("Output tensor shape: ")
            # print(output_image.shape)
            # print("Output tensor: ")
            # print(output_image)

            to_pil = transforms.ToPILImage()
            result_image = to_pil(output_image)
            result_image.save(
                os.path.join(
                    "../data/test/chunks/upscaled",
                    subdirectory,
                    "upscaled_{x}_{y}.jpg".format(x=x_str, y=y_str),
                )
            )

    path = "../data/test/chunks/upscaled"
    for subdirectory in os.listdir(path):
        stitcher.stitch_image(
            path=os.path.join(path, subdirectory),
            name="upscaled",
            name_of_save=subdirectory + ".jpg",
            save_path="../data/test/stitched",
        )
