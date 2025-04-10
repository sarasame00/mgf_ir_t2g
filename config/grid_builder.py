from itertools import product
from .presets import get_parameter_ranges
from .lat_settings import LAT_CONSTANTS
from .ss_settings import SS_CONSTANTS

def generate_lat_grid(preset, resolution):
    """
    Generate all possible combinations of parameters for a given electron configuration preset
    to be used in lattice model simulations (DysonSolver-compatible).

    Parameters:
    - preset (str): One of '3d_d1', '4d_d1', '5d_d1', '5d_d5', etc.
    - resolution (int): Number of sampled values per parameter range (used in linspace)

    Returns:
    - List of tuples formatted as:
      (T, wmax, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)
    """
    # Load model-specific parameter ranges (N, U, J, g, lbd, etc.)
    model = get_parameter_ranges(preset, resolution)

    # Load lattice-specific constants (T, wmax, w0, etc.)
    const = LAT_CONSTANTS

    # Compute cartesian product of all parameter combinations
    param_grid = product(
        [const["T"]],           # Temperature values
        [const["wmax"]],      # Max frequency (fixed)
        model["N"],           # Electron counts
        model["t"],           # Hopping
        model["U"],           # Hubbard U
        model["J"],           # Hund's coupling
        [const["Jphm"]],        # Phonon exchange
        [const["w0"]],        # Phonon frequency (fixed)
        model["g"],           # Jahn-Teller coupling
        model["lbd"],         # Spin-orbit coupling
        [const["k_sz"]],      # Momentum projection (fixed)
        [const["diis_mem"]],  # DIIS memory (fixed)
    )

    return list(param_grid)


def generate_ss_grid(preset="3d_d1", resolution=3):
    """
    Generate all possible combinations of parameters for single-site simulations.

    Parameters:
    - preset (str): One of '3d_d1', '4d_d1', '5d_d1', etc.
    - resolution (int): Number of values to sample per parameter range

    Returns:
    - List of tuples formatted as:
      (N, U, J, g, lbd, B, Qmax, size_grid)
    """
    # Load model-specific parameter ranges (N, U, J, g, lbd)
    model = get_parameter_ranges(preset, resolution)

    # Load single-site constants (Qmax, grid size)
    const = SS_CONSTANTS

    # Compute cartesian product of variable model parameters
    param_grid = product(
        model["N"],      # Electron count
        model["U"],      # Coulomb repulsion
        model["J"],      # Hund's coupling
        model["g"],      # Jahn-Teller strength
        model["lbd"],    # Spin-orbit coupling
        model["B"],
        [const["qmax"]], 
        [const["size_grid"]]
        
    )

    return list(param_grid)
