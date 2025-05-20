import os
import random
from astropy.io import fits
import numpy as np

# === CONFIG ===
PATCH_SIZE = 500
NUM_PATCHES = 100
CLEAN_IMAGE_PATH = 'f444w_finalV4.onlyPSF.fits'
NOISY_IMAGE_PATH = 'f444w_finalV4.fits'
TRUE_DIR = 'dataset/true'
IMAGE_DIR = 'dataset/image'

# === SETUP ===
os.makedirs(TRUE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# === LOAD IMAGES ===
with fits.open(CLEAN_IMAGE_PATH) as hdul:
    clean_data = hdul[0].data

with fits.open(NOISY_IMAGE_PATH) as hdul:
    noisy_data = hdul[0].data
#noisy_data = clean_data + (clean_data.max() - clean_data.min()) * np.random.normal(0, 0.01, size=clean_data.shape)
height, width = clean_data.shape

# === GET VALID PATCH COORDINATES (non-overlapping) ===
all_coords = [
    (x, y)
    for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE)
    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE)
]
if len(all_coords) < NUM_PATCHES:
    raise ValueError("Not enough non-overlapping patches available.")

random.shuffle(all_coords)
selected_coords = all_coords[:NUM_PATCHES]

# === EXTRACT AND SAVE PATCHES ===
for idx, (x, y) in enumerate(selected_coords, start=1):
    clean_patch = clean_data[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    noisy_patch = noisy_data[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    filename = f"patch_{idx:02d}_x{x}_y{y}.fits"

    fits.writeto(os.path.join(TRUE_DIR, filename), clean_patch, overwrite=True)
    fits.writeto(os.path.join(IMAGE_DIR, filename), noisy_patch, overwrite=True)

print(f"✅ {NUM_PATCHES} patches saved in '{TRUE_DIR}/' and '{IMAGE_DIR}/'")
