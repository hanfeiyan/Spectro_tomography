# Spectroscopic Tomography Reconstruction with a single sinogram

A Python toolkit for sparse spectro-tomography reconstruction using a single sinogram. It is based on a joint spectra-tomography algorithm with total-variation regularization that performs spectrum-fitting and three-dimensional reconstruction in a single step. This package provides methods for reconstructing multi-state tomographic images from sparse projection data in angle-energy space.

## Overview

For scanning x-ray microscopy, the slow speed is the downside of the technique. Particularly for spectro-tomography, where many tomograms at different energy points across the absorption edge are required, the acqusition time can be prohibitely long. To address this challenge, a joint algorithm that performs the spectrum-fitting and tomography reconstruction in a single step is developed, and it requires significantly less data for the reconstruction. This toolkit implements the method. It utilizes the ASTRA tomography library and various iterative linear solvers in scipy to perform joint reconstruction of multiple spectral states with TV (total variation) regularization.

## Requirements

- **astra-toolbox**: Tomographic reconstruction library
- **numpy**: Numerical computing
- **scipy**: Scientific computing (sparse matrices, linear algebra)
- **matplotlib**: Visualization
- **scikit-image**: Image metrics (MSE)
- **phantominator**: Phantom image generation
- **tifffile**: TIFF image I/O
- **tqdm**: Progress bars
- **threadpoolctl**: CPU thread control for fast volume reconstruction
- **cupy** (optional): GPU arrays and sparse operations for fast reconstruction

## Installation

Install the required dependencies using pip:

```bash
pip install astra-toolbox numpy scipy matplotlib scikit-image phantominator tifffile tqdm
pip install threadpoolctl
```

For GPU reconstruction, install a CuPy build compatible with your CUDA
environment (for example, `cupy-cuda12x` for CUDA 12). CPU reconstruction does
not require CuPy.

## Usage

### Basic Example

```python
from spectro_tomography import *

# input
sinogram                        # data (num_angle, num_columns)
codebook                        # known weights for states at each projection (num_angle, num_states) 
angle                           # angle list for projections

# Set reconstruction parameters
rot_cen_offset = 0              # rotation center offset
obj_size = [row, col]           # image dimensions
mu1 = 0.05                      # TV regularization weight
mu2 = 10                        # ADMM hyperparameter
max_iter = 100                  # number of iterations
method = 'gmres'                # used linear solver

# Run reconstruction
x, err = multistate_tomo_joint_TV(
    sinogram, angle, rot_cen_offset, obj_size, codebook,
    mu1, mu2, max_iter,
    method='gmres'
)

# x is a list of reconstructed images (one per spectral state)
```

### Fast Reconstruction

The optional `spectro_tomography_fast.py` module provides a prepared
`Reconstructor`, an accelerated single-slice wrapper, and a volume function.
`codebook` (also called `ref`) must have one row per measured angle and one
column per state. The default regularizer is TV; for L1 or combined TV+L1, set
`regularization="L1"` or `regularization="TV+L1"` and supply `mu3`.

#### Single slice

Prepare the projector and reconstructor once, then reuse the same instance for
repeated reconstructions with the same geometry and codebook:

```python
from spectro_tomography_fast import (
    Reconstructor,
    build_astra_projector,
    multistate_tomo_joint,
)

obj_size = (row, col)
projector = build_astra_projector(angle, obj_size, sinogram.shape[1])
engine = Reconstructor(
    projector, codebook, obj_size,
    backend="cpu", dtype="float32",
)

rec, errors, diagnostics = multistate_tomo_joint(
    sinogram, angle, rot_cen_offset, obj_size, codebook,
    mu1=0.05, mu2=1.0, max_iter=100,
    reconstructor=engine,
    algorithm="admm", method="cg",
    regularization="TV+L1", mu3=0.01,
    rtol=1e-5, maxiter=100, log_every=10,
    return_diagnostics=True,
)
# rec[state] is the reconstructed image for that state.
```

Use `backend="gpu"` and set `gpu_id` when using a compatible CuPy/CUDA
installation. When passing `reconstructor=engine`, its projector, backend,
precision, and boundary settings are reused.

To let the wrapper create the projector and reconstructor, pass
`reconstructor=None` (the default) and choose the backend settings directly:

```python
rec, errors, diagnostics = multistate_tomo_joint(
    sinogram, angle, rot_cen_offset, obj_size, codebook,
    mu1=0.02, mu2=1.0, mu3=0.2,
    max_iter=100,
    reconstructor=None,
    backend="gpu", gpu_id=0, dtype="float32",
    algorithm="admm", method="gmres",
    regularization="TV+L1",
    rtol=1e-5, maxiter=100, log_every=10,
    return_diagnostics=True,
)
```

With `reconstructor=None` and no `projector` argument, the wrapper builds the
ASTRA projector and prepares a solver for this call.

#### Volume

Volume sinograms have shape `(slices, angles, detectors)`. This example uses
CPU workers and should run from an importable Python script:

```python
from spectro_tomography_fast import build_astra_projector, reconstruct_volume

def main():
    obj_size = (row, col)
    projector = build_astra_projector(angle, obj_size, sinograms.shape[2])
    volume, info = reconstruct_volume(
        sinograms, angle, codebook, obj_size,
        projector=projector,
        backend="cpu", workers=4, threads_per_worker=1,
        algorithm="admm", method="cg",
        regularization="TV", mu1=0.05, mu2=1.0,
        max_iter=100, inner_rtol=1e-5, inner_maxiter=100,
        log_every=10, seed=12345,
        output_path="reconstructed.npy",
    )
    # volume has shape (states, slices, rows, columns).

if __name__ == "__main__":
    main()
```

For GPU volumes, use `backend="gpu"` and `gpu_ids=(...)`; workers are assigned
to the listed devices round-robin. Each worker builds its own reconstructor, so
choose the worker count with GPU memory use in mind.

More examples of fast reconstruction are provided in test_fast_reconstruction and Example_test_sample.

### Acquisition Plan Generation

Generate sampling plans for spectroscopic tomography experiments:

```python
from spectro_tomography import generate_plan

# Load or create reference spectra
ref = np.load('fe0_fe3_ref.txt')  # shape: (num_energy, num_states)

# Generate acquisition plan
angle_list, codebook, indices = generate_plan(
    ref=ref,
    angle_rng=(0, np.pi),         # angle range in radian
    num_angle=180,                # number of projections
    option='uniform-random'       # sampling strategy
)
```

**Available sampling strategies in angle-energy space:**
- `'uniform-interlaced'`: Uniform angle spacing, cyclic spectrum repetition
- `'uniform-interlaced-random'`: Uniform angle spacing, cyclic randomized spectrum selection
- `'uniform-segmented'`: Uniform angle spacing, segmented spectrum allocation
- `'uniform-random'`: Uniform angle spacing, random spectrum selection


## Core Functions

### `multistate_tomo_joint_TV()`

Main reconstruction function using ADMM optimization.

**Parameters:**
- `sinogram` (ndarray): Projection data
- `angle` (ndarray): Projection angles (radians)
- `rot_cen_offset` (float): Rotation center offset
- `obj_size` (tuple): Reconstruction image size (nx, ny)
- `ref` (ndarray): Spectral weights for each projection
- `mu1` (float): TV regularization weight
- `mu2` (float): ADMM hyperparameter
- `max_iter` (int): Maximum iterations
- `x0` (list, optional): Initial state estimates
- `nonnegative` (bool): Enforce non-negativity constraint
- `seq_save` (bool): Return full iteration sequence
- `method` (str): Linear solver ('gmres', 'cg', 'minres', etc.)

**Returns:**
- `x` (list): Reconstructed image states
- `err` (list): Objective values per iteration

### `generate_plan()`

Generate acquisition plans for spectroscopic tomography.

**Parameters:**
- `ref` (ndarray): Reference spectra (number energy, number states)
- `angle_rng` (tuple): Angle range (min, max) in radian
- `num_angle` (int): Number of projections
- `option` (str): Sampling strategy

**Returns:**
- `angle_list` (ndarray): Projection angles
- `codebook` (ndarray): known weights for each states at each projection
- `ind` (ndarray): Corresponding indices from the input reference spectra at each projection 

## Examples

The package includes three example Jupyter notebooks:

1. **Example_NMC811.ipynb** - Reconstruction of real NMC811 (lithium-ion battery cathode) material data
2. **Example_test_sample.ipynb** - Demonstrates reconstruction on a test sample
3. **Example_simulation_data.ipynb** - Shows usage with simulated phantom data

### Running Examples

```bash
jupyter notebook Example_NMC811.ipynb
```

## Data Files

Included experimental data:
- `Experimental data/NMC811/` - Real spectroscopic projection data and reference spectra for NMC811
- `Experimental data/Test object/` - Test object acquisition plan
- `fe0_fe3_ref.txt` - Reference spectra for iron (Fe) by oxidation state

## Parameters Guide

### Regularization Parameters
- **mu1** (TV weight): Controls smoothness of reconstruction. Higher values = smoother but less detailed
- **mu2** (ADMM parameter): Controls convergence behavior. Typically 1-10 for well-conditioned problems

### Rotation Center Offset
The `rot_cen_offset` parameter corrects misalignment in rotation axis:
- true_rotation_center - center_of_columns

## License

See [LICENSE](LICENSE) file for details.

## References

Yan, H., Pattammattel, A., Michelson, A., Hu, E. & Xia, K. Sparse X-ray spectro-tomography for high-sensitivity three-dimensional chemical imaging at the nanoscale. *Nature Communications* **17**, 10867 (2026). [https://doi.org/10.1038/s41467-026-77821-5](https://www.nature.com/articles/s41467-026-77821-5)

## Citation

If you use this toolkit in your research, please cite appropriately.  

## Support

For issues, questions, or contributions, please contact the development team.
