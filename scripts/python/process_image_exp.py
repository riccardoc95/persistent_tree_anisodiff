# process_image.py
# ls data/dataset/image/*.fits | sort > images_path.txt
import sys
from src.topotree import CutTree
from src.anisodiff_exp import anisotropic_graph_diffusion
from astropy.io import fits
import numpy as np
from itertools import product
import re
import os

spatial_sigma_values = [1]#[0.5, 1, 2, 3, 4, 5, 10]
intensity_sigma_values = [1e2]#[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


path = sys.argv[1]
image = fits.getdata(path)
pattern = r'\d+\.\d+/image|image'

#cut_t_values = np.linspace(2*max(image.min(), 0), 0.07, 10)

gt = fits.getdata(re.sub(pattern, 'true', path))
gt_norm = (gt - image.min()) / (image.max() - image.min())
tree = CutTree()
tree.from_image(image)
level = tree.cut()
#print(level)

best_mse = np.inf
best_reconstruct = image
best_combination = {}

back_t = -1
#for cut_t, spatial_sigma, intensity_sigma in product(cut_t_values, spatial_sigma_values, intensity_sigma_values):
for spatial_sigma, intensity_sigma in product(spatial_sigma_values, intensity_sigma_values):
    #if cut_t != back_t:
    #    tree.from_image(image)
    #    level = tree.cut(cut_t)
    #    back_t = cut_t
    new_tree, step, score = anisotropic_graph_diffusion(tree, steps=150,
                                           alpha=0.1,
                                           spatial_sigma=spatial_sigma,
                                           intensity_sigma=intensity_sigma,
                                           gth=gt_norm,)
    print(level, score, step)
    reconstructed = new_tree.to_image()
    #mse = np.mean((gt - reconstructed) ** 2)

    #if mse < best_mse:
    #    best_mse = mse
    #    best_reconstruct = reconstructed.copy()
    #    best_combination['spatial_sigma'] = spatial_sigma
    #    best_combination['intensity_sigma'] = intensity_sigma
    #    best_combination['step'] = step

    if  best_mse > score:
        best_mse = score
        best_reconstruct = reconstructed.copy()
        best_combination['spatial_sigma'] = spatial_sigma
        best_combination['intensity_sigma'] = intensity_sigma
        best_combination['step'] = step
    #elif best_mse < score:
    #    break
print(path, "->", "Best MSE:", np.mean((gt - best_reconstruct) ** 2), "Combination:", best_combination)

pattern = r'.*?/image|.*?image'

# Replace with 'true'
if "mri" in path:
    sigmas = re.findall(r'\d+\.\d+', path)
    if len(sigmas) > 0:
        sigma = sigmas[0]
        output_path = re.sub(pattern, f'outputs/mri_our_exp/{sigma}', path)
    else:
        output_path = re.sub(pattern, f'outputs/mri_our_exp', path)
else:
    output_path = re.sub(pattern, 'outputs/our_exp', path)

os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
fits.writeto(output_path, best_reconstruct, overwrite=True)
