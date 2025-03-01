#!/bin/bash

for file in *.json; do
    echo "Processing $file:"
    jq . "$file" > temp.json && mv temp.json "$file"
done
