# Persistent Trees: A Topological Tree Structure for Anisotropic Image Denoising
This project introduces a topological framework for image denoising using persistent homology. We construct a persistent tree structure from an input image and use it to guide anisotropic diffusion in a structure-aware, data-adaptive manner. The method enables unsupervised denoising by estimating diffusion parameters directly from the image's topological features.

---
## Project Folder Structure

```
.
├── peronamalik/        # Perona-Malik diffusion-related code
├── PixHomology/        # Core module for pixel-based homology computations
├── scripts/            # Scripts for automation and job management
│   ├── python/         # Python helper or experiment scripts
│   └── slurm/          # SLURM job submission scripts for HPC clusters
└── src/                # Main source code
```

## Installation of PixHomology module

To install the [`PixHomology`](https://github.com/riccardoc95/PixHomology) module, navigate to the directory and run:

```bash
cd PixHomology
pip install .
```

## Usage

1. Run a python script

```bash
python scripts/python/compute.py
```

2. Run a slurm script:
```bash
sbatch scripts/slurm/compute.sbatch
```

## Results

We evaluated our method on two challenging datasets:

* **[Knee MRI (NYU fastMRI)](https://fastmri.med.nyu.edu/)** at different noise levels (\$\sigma = 0.1, 0.2, 0.3\$)
* **[FORECAST Astronomical Image](https://www.astrodeep.eu/forecast/)**

Our goal is to assess both **topological structure recovery** and **denoising quality**.

### Tree Edit Distance (TED) to Ground Truth

This table evaluates the fidelity of the persistent tree reconstruction using TED (lower is better). We compare raw noisy trees with filtered ones using grid-search and max-jump thresholding strategies.

| **Dataset**       | **TED (Noisy)** | **TED (Filtered - Grid Search)** | **TED (Filtered - Max-Jump)** |
| ----------------- | --------------- | -------------------------------- | ----------------------------- |
| Knee MRI, σ = 0.1 | 31969 ± 180     | **16421 ± 52**                   | 16431 ± 56                    |
| Knee MRI, σ = 0.2 | 32004 ± 177     | **16424 ± 54**                   | 16435 ± 53                    |
| Knee MRI, σ = 0.3 | 32015 ± 172     | **16437 ± 59**                   | 16438 ± 59                    |
| Astronomical      | 32195 ± 376     | **16425 ± 86**                   | 19153 ± 97                    |


### Image Denoising Metrics (PSNR, SSIM, MSE)

This table compares denoising performance using three standard metrics. Our **Filtered Tree Diffusion (PF)** method achieves near-optimal results without manual tuning.

| **Dataset**           | **Method**               | **PSNR (dB)**    | **SSIM**        | **MSE**               |
| --------------------- | ------------------------ | ---------------- | --------------- | --------------------- |
| **Knee MRI, σ = 0.1** | Input                    | 20.00 ± 0.02     | 0.31 ± 0.05     | 1.00e-2 ± 4.37e-5     |
|                       | Perona Malik             | 30.35 ± 3.83     | 0.79 ± 0.11     | 2.49e-3 ± 8.65e-3     |
|                       | Unfiltered Tree Diff.    | 30.50 ± 2.40     | 0.81 ± 0.09     | 1.10e-3 ± 5.00e-4     |
|                       | **Filtered Tree Diff.**  | **31.65 ± 2.23** | **0.84 ± 0.08** | **7.91e-4 ± 4.35e-4** |
|                       | Filtered Tree Diff. (PF) | 31.25 ± 2.25     | 0.83 ± 0.08     | 8.16e-4 ± 4.52e-4     |
| **Knee MRI, σ = 0.2** | Input                    | 13.98 ± 0.02     | 0.15 ± 0.02     | 4.00e-2 ± 1.77e-4     |
|                       | Perona Malik             | 28.18 ± 1.45     | 0.81 ± 0.05     | 1.61e-3 ± 5.71e-4     |
|                       | Unfiltered Tree Diff.    | 28.35 ± 1.55     | 0.84 ± 0.06     | 1.55e-3 ± 5.60e-4     |
|                       | **Filtered Tree Diff.**  | **29.10 ± 1.58** | **0.87 ± 0.06** | **1.29e-3 ± 5.40e-4** |
|                       | Filtered Tree Diff. (PF) | 28.95 ± 1.58     | 0.86 ± 0.06     | 1.35e-3 ± 5.42e-4     |
| **Knee MRI, σ = 0.3** | Input                    | 10.46 ± 0.02     | 0.10 ± 0.01     | 9.00e-2 ± 4.04e-4     |
|                       | Perona Malik             | 26.21 ± 1.17     | 0.81 ± 0.04     | 2.49e-3 ± 6.94e-4     |
|                       | Unfiltered Tree Diff.    | 26.80 ± 1.20     | 0.86 ± 0.05     | 2.15e-3 ± 5.90e-4     |
|                       | **Filtered Tree Diff.**  | **27.35 ± 1.15** | **0.89 ± 0.04** | **1.95e-3 ± 5.72e-4** |
|                       | Filtered Tree Diff. (PF) | 27.20 ± 1.16     | 0.88 ± 0.04     | 2.06e-3 ± 5.75e-4     |
| **Astronomical**      | Input                    | 43.47 ± 9.41     | 0.84 ± 0.26     | 6.64e-8 ± 9.28e-9     |
|                       | Perona Malik             | 55.44 ± 6.89     | 0.97 ± 0.07     | 5.40e-9 ± 3.69e-9     |
|                       | Unfiltered Tree Diff.    | 55.10 ± 6.85     | 0.97 ± 0.06     | 5.10e-9 ± 3.50e-9     |
|                       | **Filtered Tree Diff.**  | **56.05 ± 6.79** | **0.98 ± 0.06** | **4.66e-9 ± 3.05e-9** |
|                       | Filtered Tree Diff. (PF) | 55.95 ± 6.81     | 0.98 ± 0.06     | 4.79e-9 ± 3.12e-9     |



## License

This project is licensed under the MIT License. See LICENSE for more details.


## Contact
For questions or collaborations:

- Author: Riccardo Ceccaroni
- Email: riccardo.ceccaroni@uniroma1.it

