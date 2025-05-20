from src.topotree import Tree, CutTree
from src.anisodiff import anisotropic_graph_diffusion

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt


from glob import glob
from tqdm import tqdm
import sys
import re
import os

def tree_edit_distance(tree1, tree2):
    from functools import lru_cache

    def get_children(tree, node_id):
        return tree.get_successors().get(node_id, [])

    def get_label(tree, node_id):
        return tree.node_labels[node_id]

    @lru_cache(maxsize=None)
    def distance(n1, n2):
        children1 = get_children(tree1, n1)
        children2 = get_children(tree2, n2)

        m = len(children1)
        n = len(children2)

        # Initialize DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + subtree_cost(tree1, children1[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + subtree_cost(tree2, children2[j - 1])

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = distance(children1[i - 1], children2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + subtree_cost(tree1, children1[i - 1]),  # delete
                    dp[i][j - 1] + subtree_cost(tree2, children2[j - 1]),  # insert
                    dp[i - 1][j - 1] + cost  # relabel (or match)
                )

        label_cost = 0 if get_label(tree1, n1) == get_label(tree2, n2) else 1
        return dp[m][n] + label_cost

    def subtree_cost(tree, node_id):
        """Cost of deleting/inserting an entire subtree rooted at node_id."""
        total = 1  # count current node
        for child in get_children(tree, node_id):
            total += subtree_cost(tree, child)
        return total

    return distance(tree1.root, tree2.root)


path = sys.argv[1]
image = fits.getdata(path)[:128,:128]
pattern = r'\d+\.\d+/image|image'
gt = fits.getdata(re.sub(pattern, 'true', path))[:128,:128]

true_tree = Tree()
input_tree = Tree()
cut_tree = CutTree()

true_tree.from_image(gt)
input_tree.from_image(image)
cut_tree.from_image(image)
cut_tree.cut()

dist_input_true = tree_edit_distance(input_tree, true_tree)
dist_cut_true = tree_edit_distance(cut_tree, true_tree)

print(path)
print("Input/True Distance:", dist_input_true)
print("Cut/True Distance:", dist_cut_true)


pattern = r'.*?/image|.*?image'

# Replace with 'true'
if "mri" in path:
    sigmas = re.findall(r'\d+\.\d+', path)
    if len(sigmas) > 0:
        sigma = sigmas[0]
        output_path = re.sub(pattern, f'outputs/mri_tree_distance/{sigma}', path)
    else:
        output_path = re.sub(pattern, f'outputs/mri_tree_distance', path)
else:
    output_path = re.sub(pattern, 'outputs/tree_distance', path)

os.makedirs("".join(output_path.split("/")[:-1]), exist_ok=True)

np.savez(output_path.replace(".fits", ".npz"),
         dist_input_true=dist_input_true,
         dist_cut_true=dist_cut_true)
