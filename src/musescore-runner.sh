#!/usr/bin/env sh

# takes a file path, and converts the file into data in a new directory
# an individual directory contains a .xml and a folder of pngs

if ! command -v mscore &> /dev/null; then
    echo "Musescore is not installed."
    exit 1
fi

echo "Is this even working?"

file_path=$1
output_xml=$2/xml.xml
output_png=$2/png/png.png
mkdir "$2"
mkdir "$2"/png

echo "file path: $file_path"
echo "xml path: $output_xml"
echo "png path: $output_png"

mscore "$file_path" -o "$output_png"
echo "made the png"

mscore "$file_path" -o "$output_xml"
echo "made the xml"
