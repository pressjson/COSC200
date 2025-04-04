#!/usr/bin/env python3


import os
import re
from PIL import Image

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


def stitch_images(name, directory, delta=settings.DELTA):
    """stitches an image of the form %name%_%x_offset%_%y_offset%.jpg
    args:
        name: the name of the image to be stitched within the directory
        directory: the directory containing the images to be stitched
        delta: how big each jump in images is"""
    pattern = r"(\w+)_(\d+)_(\d+)\.jpg"
    x_max = 0
    y_max = 0
    for file in os.listdir(directory):
        if not file.__contains__(name):
            continue
        match = re.match(pattern, file)
        if match:
            name, x_str, y_str = match.groups()
            x = int(x_str)
            y = int(y_str)
            if x >= x_max:
                x_max = x
            if y >= y_max:
                y_max = y

    result_image = Image.new("RGB", (x_max + delta + 1, y_max + delta + 1))
    x = 0
    y = 0
    while x <= x_max:
        y = 0
        while y <= y_max:
            image = Image.open(
                os.path.join(directory, f"{name}_{x}_{y}.jpg"),
            )

            result_image.paste(image, (x, y))
            y = y + delta
        x = x + delta

    return result_image


def stitch_image(
    path="../data/chunks",
    name="hq",
    name_of_save="stitched.jpg",
    save_path="../data/stitched",
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result = stitch_images(name, path)
    result.save(os.path.join(save_path, name_of_save))
