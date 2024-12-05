#!/usr/bin/env python3
"""Loads the data given an integer representing the directory number.
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy
import os

import partitura


def loader(path: str):
    """Private method to convert a path into a tensor."""
    print("Loading " + path)
    if path.endswith(".png"):
        image = Image.open(path)
        # transform = transforms.ToTensor()
        # tensor = transform(image)
        arr = numpy.array(image)
        tensor = torch.from_numpy(arr)
        return tensor

    if path.endswith(".xml"):
        score = partitura.load_musicxml(path)
        roll = partitura.utils.compute_pitch_class_pianoroll(score, True, "auto")
        tensor = torch.from_numpy(roll)
        return tensor

    raise Exception(
        "Something went wrong trying to load " + path + ":\r\tMust be a png or xml"
    )


def make_image_tensor(i: int):
    path = "../data/" + str(i) + "/png/"
    tensor = loader(path + "png-1.png")

    for infile in os.listdir(path):
        if infile == "png-1.png":
            continue
        path_to_file = path + infile
        image_tensor = loader(path_to_file)
        tensor = torch.cat((tensor, image_tensor), 0)

    return tensor


def make_xml_tensor(i: int):
    path = "../data/" + str(i) + "/xml.xml"
    return loader(path)
