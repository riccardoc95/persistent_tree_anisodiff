#!/bin/bash
#SBATCH --job-name=diffusion_exp
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=0-99         # Adapt this to your number of images
#SBATCH --nodelist=cn1a,cn1b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Get the list of files
FILELIST=$(ls data/dataset/image/*.fits | sort)
FILE=$(echo "$FILELIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

python -m scripts.python.process_image_exp "$FILE"