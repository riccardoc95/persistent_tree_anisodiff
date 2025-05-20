import numpy as np
import os
import glob
from astropy.io import fits
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import math

INPUT_FOLDER = "data/dataset/image"
TRUE_FOLDER = "data/dataset/true"
PM_FOLDER = "outputs/peronamalik"
OUR_FOLDER = "outputs/our_exp"

FILES = [os.path.basename(file) for file in glob.glob(OUR_FOLDER + "/*.fits")]

def print_metrics(metrics_dict):
    print("-" * 50)
    print(f"{'Metric':<10} | {'Mean':^18} | {'Std Dev':^18}")
    print("-" * 50)
    for key, values in metrics_dict.items():
        mean = values['mean']
        std = values['std']
        if key == "MSE":
            print(f"{key:<10} | {mean:^18.2e} | {std:^18.2e}")
        else:
            print(f"{key:<10} | {mean:^18.2f} | {std:^18.2f}")
    print("-" * 50)

def compute_mean_and_std(results):
    """
    Given a list of dictionaries with metrics (MSE, PSNR, SSIM), compute the mean and standard deviation for each metric.

    Parameters:
    results (list of dicts): List of dictionaries where each dictionary contains metrics for one image.

    Returns:
    dict: Dictionary with mean and std for each metric (MSE, PSNR, SSIM).
    """
    metrics = ['MSE', 'PSNR', 'SSIM']

    # Extract values for each metric across all results
    all_metrics = {metric: [] for metric in metrics}

    for result in results:
        for metric in metrics:
            all_metrics[metric].append(result[metric])

    # Compute mean and std for each metric
    metrics_mean_std = {}
    for metric in metrics:
        values = all_metrics[metric]
        metrics_mean_std[metric] = {
            'mean': np.mean(values).item(),
            'std': np.std(values).item()
        }

    return metrics_mean_std

def denoising_metrics(ground_truth, denoised_image, max_, min_):
    """
    Compute the principal metrics for denoising image comparison.

    Parameters:
    ground_truth (ndarray): The original clean image.
    denoised_image (ndarray): The denoised image to compare.

    Returns:
    dict: Dictionary with MSE, PSNR, and SSIM.
    """

    # Ensure images are in the same shape
    assert ground_truth.shape == denoised_image.shape, "The images must have the same dimensions"

    # Compute MSE (Mean Squared Error)
    mse = np.mean((ground_truth - denoised_image) ** 2)

    # Compute PSNR (Peak Signal to Noise Ratio)
    if mse == 0:
        psnr = float('inf')  # If images are identical, PSNR is infinity
    else:
        max_pixel = ground_truth.max()  # Assuming 8-bit images, max pixel value is 255
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    # Compute SSIM (Structural Similarity Index)
    ssim_value = ssim(ground_truth, denoised_image, data_range=max_ - min_)

    return {
        'MSE': mse.item(),
        'PSNR': psnr,
        'SSIM': ssim_value.item()
    }

results_input = []
results_pm = []
results_our = []
for file in FILES:#["patch_01_x26500_y0.fits"]:
    input = fits.getdata(os.path.join(INPUT_FOLDER, file))
    true = fits.getdata(os.path.join(TRUE_FOLDER, file))
    pm = fits.getdata(os.path.join(PM_FOLDER, file))
    our = fits.getdata(os.path.join(OUR_FOLDER, file))

    max_, min_ = input.max(), input.min()

    results_input.append(denoising_metrics(true, input, max_, min_))
    results_pm.append(denoising_metrics(true, pm, max_, min_))
    results_our.append(denoising_metrics(true, our, max_, min_))


input_metrics_stats = compute_mean_and_std(results_input)
pm_metrics_stats = compute_mean_and_std(results_pm)
our_metrics_stats = compute_mean_and_std(results_our)

# Print the results
print("Input Image Metrics:")
print_metrics(input_metrics_stats)

print("\nPM Image Metrics:")
print_metrics(pm_metrics_stats)

print("\nOur Method Image Metrics:")
print_metrics(our_metrics_stats)


def top_n_images_where_our_is_better(results_pm, results_our, filenames, metric='SSIM', n=10):
    """
    Prints the top N image filenames where OUR method outperforms PM the most for a given metric.

    Parameters:
    - results_pm: list of dicts with PM metrics
    - results_our: list of dicts with OUR metrics
    - filenames: list of image filenames
    - metric: 'SSIM', 'PSNR', or 'MSE'
    - n: number of top images to show
    """
    diffs = []
    for i in range(len(filenames)):
        pm_val = results_pm[i][metric]
        our_val = results_our[i][metric]
        if metric in ['PSNR', 'SSIM']:
            diff = our_val - pm_val  # higher is better
        else:  # MSE
            diff = pm_val - our_val  # lower is better
        diffs.append((filenames[i], diff))

    # Sort by improvement (descending)
    top = sorted(diffs, key=lambda x: x[1], reverse=True)[:n]

    print(f"\nTop {n} images where OUR method outperforms PM the most in {metric}:")
    for fname, diff in top:
        print(f"{fname}: Δ{metric} = {diff:.4f}")


top_n_images_where_our_is_better(results_pm, results_our, FILES, metric='SSIM', n=10)
top_n_images_where_our_is_better(results_pm, results_our, FILES, metric='MSE', n=10)
top_n_images_where_our_is_better(results_pm, results_our, FILES, metric='PSNR', n=10)

