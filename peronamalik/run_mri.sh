#!/bin/bash

IMAGE_DIR="../data/mri_dataset/noisy_s_0.3"
TRUE_DIR="../data/mri_dataset/clean"
G_VALUE=2

for input_file in "$IMAGE_DIR"/*.fits; do
    filename=$(basename "$input_file")
    image_path="$IMAGE_DIR/$filename"
    true_path="$TRUE_DIR/$filename"

    echo "Running on: $filename"
    ./pm "$image_path" "$G_VALUE" "$true_path"
done