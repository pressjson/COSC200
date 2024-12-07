#!/usr/bin/env sh

# takes a file path, and converts the file into data in a new directory

if ! command -v mscore &> /dev/null; then
    echo "Musescore is not installed. Install Musescore and try again."
    exit 1
fi

file_path=$1
hq_png="../data/$2/hq/png.png"
lq_png="../data/$2/lq/png.png"
mkdir "../data/"$2
mkdir "../data/$2/hq"
mkdir "../data/$2/lq"

mscore "$file_path" -o "$hq_png"
mscore "$file_path" -o "$lq_png"
