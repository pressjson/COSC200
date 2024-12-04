#!/usr/bin/env python3

import os
import subprocess

print("This will run through an example, preparing the Moonlight Sonata in a newly created /data directory.")

url = "https://musescore.com/user/41399859/scores/12043831"

script1 = "./librescore-runner.sh"
script2 = "./musescore-runner.sh"

subprocess.run([script1, url])

cwd = os.getcwd()
root = os.path.dirname(cwd)
data_path = root + "/data"

i = 0
for infile in
