# Enhanced 3D Gaussian Splatting for Fisheye Cameras

This repository provides the official implementation of the paper:

**Enhanced 3D Gaussian Splatting for Fisheye Cameras:  
Jacobian Analysis and Dynamic Masking**

The manuscript corresponding to this code is currently **submitted to _The Visual Computer_**.

---

## 1. Setup

We recommend using Conda to create an isolated environment.

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```
To train a 3D Gaussian Splatting model on a fisheye dataset, run:
## 2. Training
```bash
python train.py \
    -s <path to your dataset> \
    -m <path to model to be trained>
```
<path to your dataset>: Path to the preprocessed fisheye dataset

<path to model to be trained>: Output directory where checkpoints and logs will be saved

The training pipeline integrates a fisheye camera projection model with analytic Jacobians and dynamic masking to improve stability and rendering quality near image boundaries.
## 3. Rendering
```bash
python render.py \
    -s <path to your dataset> \
    -m <path to model to be trained>
```
The rendered results will be saved to the corresponding output directory.
This script supports fisheye camera geometry and applies the same projection and masking strategies used during training.
