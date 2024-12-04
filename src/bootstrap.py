#!/usr/bin/env python3

import os
import subprocess

print(
    "This will run through an example, preparing the Moonlight Sonata in a newly created /data directory."
)

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
        output_path = "../data/" + str(i)
        print(file_path)

        subprocess.run(["bash", musescore_runner, file_path, output_path])
        i = i + 1
