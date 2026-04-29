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

## Installation

Install the required dependencies using pip:

```bash
pip install astra-toolbox numpy scipy matplotlib scikit-image phantominator tifffile tqdm
```

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

Hanfei Yan, Ajith Pattammattel, Aaron Michelson et al. Sparse X-ray Spectro-Tomography for High-Sensitivity Three-Dimensional Chemical Imaging at the Nanoscale, 03 June 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6659989/v1]

## Citation

If you use this toolkit in your research, please cite appropriately.  

## Support

For issues, questions, or contributions, please contact the development team.
