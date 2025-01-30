#!/usr/bin/env python3

import os
import subprocess

def get_imslp_files():
    url_file = "imslp_urls.txt"

    subprocess.run(["mkdir", "../data"])
    subprocess.run(["mkdir", "../data/pdf_files"])

    with open(url_file, "r") as infile:
        i = 0
        for url in infile:
            url = url.rstrip()
            command = [
                "curl",
                url,
                "--output",
                "../data/pdf_files/" + str(i),
            ]
            print(command)
            subprocess.run(command)
            i = i + 1
