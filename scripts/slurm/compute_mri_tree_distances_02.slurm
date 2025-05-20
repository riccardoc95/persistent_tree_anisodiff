#!/bin/bash
#SBATCH --job-name=mri_02_tree_distance
#SBATCH --output=logs/mri_02_tree_distance_%A_%a.out
#SBATCH --error=logs/mri_02_tree_distance_%A_%a.err
#SBATCH --array=0-350       # Adjust based on the number of images
#SBATCH --nodelist=cn1a,cn1b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00    # Adjust as needed
#SBATCH --mem=16G

# Get the list of files
FILELIST=$(ls data/mri_dataset/0.2/image/*.fits | sort)
FILE=$(echo "$FILELIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Processing $FILE"
python -m scripts.python.compute_tree_distance "$FILE"
