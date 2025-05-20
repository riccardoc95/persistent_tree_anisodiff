import numpy as np
from glob import glob

input_distances = []
cut_distances = []

for path in sorted(glob("outputs/best_tree_distance/*.npz")):
    data = np.load(path)
    input_distances.append(data["dist_input_true"])
    cut_distances.append(data["dist_cut_true"])

input_distances = np.array(input_distances)
cut_distances = np.array(cut_distances)

print("Input/True Distance:")
print("  Mean:", np.mean(input_distances))
print("  Std:", np.std(input_distances))

print("\nCut/True Distance:")
print("  Mean:", np.mean(cut_distances))
print("  Std:", np.std(cut_distances))