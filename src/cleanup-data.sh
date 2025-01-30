#!/usr/bin/env sh

find "../data" -maxdepth 1 -type f -name "*.mid" -exec rm -f {} \;
