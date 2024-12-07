#!/usr/bin/env python3

import os
import subprocess
from PIL import Image, ImageFilter


def degrade_images(path: str):
    print("Degrading " + path)
    for infile in os.listdir(path):
        print(infile)
        image = Image.open(path + "/" + infile)
        degraded_image = image.filter(ImageFilter.BLUR)

        degraded_image.save(path + "/" + infile)


librescore_runner = "librescore-runner.sh"
musescore_runner = "musescore-runner.sh"

# run librescore-runner on as many files inputted in the JSON that does not exist yet

with open("url.txt", "r") as infile:
    for url in infile:
        print(url)
        subprocess.run(["bash", librescore_runner, url])

# run musescore-runner for each valid mid file
i = 0
for infile in os.listdir("../data"):
    if infile.endswith(".mid"):
        file_path = "../data/" + infile
        print(file_path)

        subprocess.run(["bash", musescore_runner, file_path, str(i)])
        i = i + 1

subprocess.run(["bash", "cleanup-data.sh"])

i = 0
for infile in os.listdir("../data"):
    degrade_images("../data/" + str(i) + "/lq")
    i = i + 1
