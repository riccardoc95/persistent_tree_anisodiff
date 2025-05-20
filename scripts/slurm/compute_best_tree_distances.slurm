#!/bin/bash
#SBATCH --job-name=tree_distance
#SBATCH --output=logs/tree_distance_%A_%a.out
#SBATCH --error=logs/tree_distance_%A_%a.err
#SBATCH --array=0-99       # Adjust based on the number of images
#SBATCH --nodelist=cn1a,cn1b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00    # Adjust as needed
#SBATCH --mem=16G

# Get the list of files
FILELIST=$(ls data/dataset/image/*.fits | sort)
FILE=$(echo "$FILELIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Processing $FILE"
python -m scripts.python.compute_best_tree_distance "$FILE"
