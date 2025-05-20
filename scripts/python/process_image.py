# process_image.py
# ls data/dataset/image/*.fits | sort > images_path.txt
import sys
from src.topotree import CutTree
from src.anisodiff import anisotropic_graph_diffusion
from astropy.io import fits
import numpy as np
from itertools import product
import re
import os

spatial_sigma_values = [0.5, 1, 2, 3, 4, 5, 10]  
intensity_sigma_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

path = sys.argv[1]
image = fits.getdata(path)
pattern = r'\d+\.\d+/image|image'

gt = fits.getdata(re.sub(pattern, 'true', path))
gt_norm = (gt - image.min()) / (image.max() - image.min())
tree = CutTree()
tree.from_image(image)
level = tree.cut()

best_mse = np.inf
best_reconstruct = image
best_combination = {}

for spatial_sigma, intensity_sigma in product(spatial_sigma_values, intensity_sigma_values):
    new_tree = anisotropic_graph_diffusion(tree, steps=50,
                                           alpha=0.1,
                                           spatial_sigma=spatial_sigma,
                                           intensity_sigma=intensity_sigma,
                                           gth=gt_norm)
    reconstructed = new_tree.to_image()
    mse = np.mean((gt - reconstructed) ** 2)

    if mse < best_mse:
        best_mse = mse
        best_reconstruct = reconstructed.copy()
        best_combination['spatial_sigma'] = spatial_sigma
        best_combination['intensity_sigma'] = intensity_sigma

print(path, "->", "Best MSE:", best_mse, "Combination:", best_combination)

pattern = r'.*?/image|.*?image'

# Replace with 'true'
if "mri" in path:
    sigmas = re.findall(r'\d+\.\d+', path)
    if len(sigmas) > 0:
        sigma = sigmas[0]
        output_path = re.sub(pattern, f'outputs/mri_our/{sigma}', path)
    else:
        output_path = re.sub(pattern, f'outputs/mri_our', path)
else:
    output_path = re.sub(pattern, 'outputs/our', path)

os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
fits.writeto(output_path, best_reconstruct, overwrite=True)
