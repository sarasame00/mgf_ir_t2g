
# Matsubara Green's functions solver for t₂g systems

This repository provides a Python module to solve Matsubara Green's functions using the sparse-intermediate representation (sparse-IR) method in t₂g systems. It includes implementations for both lattice models (Matsubara formalism) and single-site models. Features include optional Google Drive synchronization and support for parallel execution.

---

## Getting started

### Prerequisites
- Python 3.x
- Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Running simulations

Both the lattice and single-site models support two modes of execution:
- **Batch mode** using parameter presets (with optional resolution for ranges)
- **Single simulation** mode with explicitly defined parameters

### Preset simulations

To run all parameter combinations defined by a preset:

#### Lattice model
```bash
python t2g_jt_soc/lat_main.py
```
####  Single-site model
```bash
python single_site/ss_main.py
```

Steps:

1. Choose a preset from `config/presets.py` (e.g., `3d_d1_r1`, `5d_d5_r2`, etc.).
2. If the preset defines ranges, set a `RESOLUTION` value to control grid density.
3. The script will generate and run all combinations of the parameter grid.

### Single simulations
For direct control over parameters, you can call the runner functions directly:
#### Lattice model
```python
val = (T, wmax, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)
run_single_simulation(val)
```
#### Single-site model
```python
val = (N, U, J, g, lbd, B, Qmax, size_grid)
run_ss_simulation(val)
```
Set the `upload_to_drive` flag if you want to enable/disable Google Drive sync.
   > Simulation results are saved with a timestamp as the filename. A CSV file logs all simulations, including parameters and corresponding timestamps. Separate CSV files are maintained for lattice and single-site simulations. These CSV files are stored both locally and on Google Drive (if synchronization is enabled).

## Configuration
Configuration files are located in the `config/` directory:
- `grid_builder.py`: Generates all possible combinations of parameters for a given preset.
- `lat_settings.py` and `ss_settings.py`: Define simulation constants and configure output data files.
- `presets.py`: Contains predefined parameter sets for different ion types. To add a new preset, include it in the presets dictionary:
```python
"new_preset": {
    "N": [values],
    "t": [values],
    "U": [values],
    "J": [values],
    "lbd": [values],
    "g": [values],
    "B": [values],
}
```
## Google Drive integration
The module includes optional Google Drive synchronization:
- Automatically uploads simulation results to a specified Google Drive folder.
- Skips simulations that have already been uploaded.
- Controlled via the upload_to_drive flag in runner functions.

### Setup
Place your Google service account JSON file in the root directory of the repository.

Configure the Google Drive folder ID in `drive_utils.py` and update folder IDs in `lat_settings.py` and `ss_settings.py`.

## Visualization
- **Single-site mmodel**: Use single-ss-plots.py for plotting results.
- **Lattice model**: Plotting functions are available in `analysis/visualize.py`, and scripts for generating plots are located in the `scripts/` directory.

