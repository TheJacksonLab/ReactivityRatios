#!/bin/bash

# Check if the input file exists
if [ ! -f "./Handbook_res.md" ]; then
    echo "Error: ./Handbook_res.md not found"
    exit 1
fi

# Filter lines that begin with "|" and save to a temporary file
grep "^|" "./Handbook_res.md" > "./Handbook_res_filtered.md"

# # Replace the original file with the filtered content
# mv "./Handbook_res_filtered.md" "./Handbook_res.md"

echo "Filtered ./Handbook_res.md - kept only lines beginning with '|'"