#!/bin/bash
#SBATCH --job-name=mri_01_diffusion
#SBATCH --output=logs/mri_01_diffusion_%A_%a.out
#SBATCH --error=logs/mri_01_diffusion_%A_%a.err
#SBATCH --array=0-350         # Adapt this to your number of images
#SBATCH --nodelist=cn1a,cn1b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Get the list of files
FILELIST=$(ls data/mri_dataset/0.1/image/*.fits | sort)
FILE=$(echo "$FILELIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

python -m scripts.python.process_image "$FILE"