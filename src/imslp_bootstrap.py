#!/usr/bin/env python3

import os
import subprocess
import time


def get_imslp_files() -> None:
    """
    sets up ../data and curls files from IMSLP (see imslp_urls.txt for the full list of files)

    for each pdf file in imslp_urls.txt, a pdf file is downloaded into ../data/pdf_files

    dependencies:
        curl

    """

    url_file = "imslp_urls.txt"

    subprocess.run(["mkdir", "../data"])
    subprocess.run(["mkdir", "../data/pdf_files"])

    with open(url_file, "r") as infile:

        i = 0
        for url in infile:
            url = url.strip()
            time.sleep(1)  # so IMSLP doesn't get mad
            command = [
                "curl",
                url,
                "--output",
                "../data/pdf_files/" + str(i) + ".pdf",
            ]
            print(command)
            subprocess.run(command)
            i = i + 1


def make_images(
    input_path="../data/pdf_files/", output_path="../data/images", output_name="0.jpg"
):
    """
    converts the files input_path into a series of jpg files in output_path

    dependencies:
        imagemagick

    args:
        input_path: the path of the pdf file, with file extension
        output_path: the place where the images are supposed to be saved
        output_name: name of the file to be outputted, with file extension

    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    command = [
        "magick",
        "-density",
        "300",
        input_path,
        os.path.join(output_path, output_name),
    ]
    print(command)
    subprocess.run(command)


def make_image_pairs(cleanup=True) -> None:
    """
    converts the jpgs in ../data/images into high quality/low quality image pairs in ../data/images/i

    low quality images are generated by shrinking the size to 25% and using jpeg artifacting with a
    compression quality of 15, and then rescaling the shrinked image to its original size

    dependencies:
        imagemagick
        the execution of make_images()

    args:
        cleanup (bool): toggles whether or not to remove the pdf images from ../data/images

    @TODO: clean this up
    """
    i = 0
    path = "../data/pdf_files"
    for pdf_file in os.listdir("../data/pdf_files"):
        make_images(
            input_path=os.path.join(path, pdf_file), output_name=str(i) + ".jpg"
        )
        i = i + 1

    i = 0

    for image in os.listdir("../data/images"):
        if not image.endswith(".jpg"):
            continue
        path = "../data/images/" + str(i)
        subprocess.run(["mkdir", path])
        command = [
            "magick",
            "-quality",
            "100",
            "../data/images/" + image,
            path + "/hq.jpg",
        ]
        print(command)
        subprocess.run(command)

        command = [
            "magick",
            "../data/images/" + image,
            "-resize",
            "25%",
            "-quality",
            "15",
            path + "/lq.jpg",
        ]
        print(command)
        subprocess.run(command)

        command = [
            "magick",
            path + "/lq.jpg",
            "-resize",
            "400%",
            "-quality",
            "100",
            path + "/lq.jpg",
        ]
        print(command)
        subprocess.run(command)

        if cleanup:
            subprocess.run(["rm", "../data/images/" + image])

        i = i + 1
