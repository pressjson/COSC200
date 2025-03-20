#!/usr/bin/env python3

import os
from PIL import Image


def concat_horizontal(image_1, image_2) -> Image:
    combined = Image.new("RGB", (image_1.width + image_2.width, image_1.height))
    combined.paste(image_1, (0, 0))
    combined.paste(image_2, (image_1.width, 0))
    return combined


def concat_vertical(image_1, image_2) -> Image:
    combined = Image.new("RGB", (image_1.width, image_2.width + image_1.height))
    combined.paste(image_1, (0, 0))
    combined.paste(image_2, (0, image_1.width))
    return combined


def stitch(directory):
    """
    files should be named %name%_%x_offset%_%y_offset%.jpg within the directory

    @TODO: write this
    """
