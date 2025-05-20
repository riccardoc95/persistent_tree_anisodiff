#!/bin/bash

IMAGE_DIR="../data/dataset/image"
TRUE_DIR="../data/dataset/true"
K_VALUE=0.1
G_VALUE=2

for input_file in "$IMAGE_DIR"/*.fits; do
    filename=$(basename "$input_file")
    image_path="$IMAGE_DIR/$filename"
    true_path="$TRUE_DIR/$filename"

    echo "Running on: $filename"
    ./pm "$image_path" "$G_VALUE" "$true_path"
done