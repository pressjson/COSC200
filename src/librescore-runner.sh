#!/bin/bash

# takes a musescore url and automatically downloads it to `data`
# dependencies: expect

# Check if `expect` is installed
if ! command -v npx &>/dev/null; then
  echo "npx is not installed. Install Node.js or npx and try again."
  exit 1
fi

# URL from the command line argument
URL=$1

# one step back from pwd to write to data
if ! [ -d ../data ]; then
  mkdir ../data
fi
data_dir=$(dirname $(pwd))/data

# Use `expect` to automate the interaction with `dl-librescore`
expect <<EOF
set timeout 2
spawn npx dl-librescore
expect "(starts with https://musescore.com/ or is a path) usually Ctrl+Shift+V to paste"
send "$URL\r"
expect "Continue? (Y/n)"
send "Y\r"
expect "Filetype Selection (Press <space> to select, <a> to toggle all, <i> to invert selection, and <enter> to proceed)"
send " \r"
expect "Output Directory"
send "$data_dir\r"
set timeout -1
expect eof
EOF
