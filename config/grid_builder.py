from itertools import product
from .presets import get_parameter_ranges
from .lat_settings import LAT_CONSTANTS
from .ss_settings import SS_CONSTANTS

def generate_lat_grid(preset, resolution):
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
    const = LAT_CONSTANTS

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


def generate_ss_grid(preset="3d_d1", resolution=3):
    """
    Generate all possible combinations of single-site parameters.

    Parameters:
    - preset (str): One of '3d_d1', '4d_d1', '5d_d1', etc.
    - resolution (int): Number of values to sample per parameter range

    Returns:
    - List of tuples:
      (N, U, J, g, lbd, B, Qmax, size_grid)
    """
    model = get_parameter_ranges(preset, resolution)
    const = SS_CONSTANTS

    # Create cartesian product of model parameters
    param_grid = product(
        model["N"],
        model["U"],
        model["J"],
        model["g"],
        model["lbd"],
    )

    # Append fixed constants to each parameter set
    grid = [
        (N, U, J, g, lbd, const["B"], const["qmax"], const["size_grid"])
        for (N, U, J, g, lbd) in param_grid
    ]

    return grid
