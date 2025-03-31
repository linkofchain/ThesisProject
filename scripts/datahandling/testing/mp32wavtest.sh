#!/bin/bash

input_dir="../../../data/teanglann/audio"
output_dir="${input_dir}/wav_files"

for file in "$input_dir"/*.mp3; do
    filename=$(basename "$file" .mp3)

echo "$filename"    
done