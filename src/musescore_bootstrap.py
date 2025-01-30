#!/usr/bin/env python3

import os
import subprocess
from PIL import Image, ImageFilter

# from skimage.util import random_noise


def noisify():
    # return lambda img: random_noise(img, mode="gaussian")
    raise Exception("noisify() is not implemented yet lmao, skill issue.")


def degrade_images(path: str):
    print("Degrading " + path)
    for infile in os.listdir(path):
        print(infile)
        image = Image.open(path + "/" + infile)
        degraded_image = image.filter(ImageFilter.BLUR).filter(ImageFilter.BoxBlur(2))

        degraded_image.save(path + "/" + infile)


if __name__ == "__main__":
    shell = "bash"
    librescore_runner = "librescore-runner.sh"
    musescore_runner = "musescore-runner.sh"
    cleanup_data = "cleanup-data.sh"
    url_path = "url.txt"

    # run librescore-runner on as many files inputted in the JSON that does not exist yet

    with open(url_path, "r") as infile:
        for url in infile:
            print(url)
            subprocess.run([shell, librescore_runner, url])
            # if result.check_returncode():
            # raise Exception("Hmm, something went wrong. Check the console") #

    i = 0
    for infile in os.listdir("../data"):
        if infile.endswith(".mid"):
            file_path = "../data/" + infile
            print(file_path)

            subprocess.run([shell, musescore_runner, file_path, str(i)])
            # if result.check_returncode():
            # raise Exception("Hmm, something went wrong. Check the console")

            i = i + 1

    subprocess.run([shell, cleanup_data])
    # if result.check_returncode():
    # raise Exception("Hmm, something went wrong. Check the console")

    i = 0
    for infile in os.listdir("../data"):
        degrade_images("../data/" + str(i) + "/lq")
        i = i + 1
