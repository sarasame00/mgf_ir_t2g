from itertools import product
from config.presets import get_parameter_ranges
from config.settings import SIMULATION_CONSTANTS

def generate_grid(preset="3d_d1", resolution=3):
    """
    Generate all possible combinations of parameters for a given electron configuration preset.

    Parameters:
    - preset (str): One of '3d_d1', '4d_d1', '5d_d1', '5d_d5', etc.
    - resolution (int): Number of values per range parameter (for np.linspace)

    Returns:
    - List of tuples formatted for DysonSolver:
      (T, wmax, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)
    """
    model = get_parameter_ranges(preset, resolution)
    const = SIMULATION_CONSTANTS

    # Product of all parameter combinations
    param_grid = product(
        const["T"],
        [const["wmax"]],
        model["N"],
        model["t"],
        model["U"],
        model["J"],
        model["Jphm"],
        [const["w0"]],
        model["g"],
        model["lbd"],
        [const["k_sz"]],
        [const["diis_mem"]],
    )

    return list(param_grid)
