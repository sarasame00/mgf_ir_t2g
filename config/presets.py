import numpy as np

def get_parameter_ranges(preset: str, resolution: int = 3):
    """
    Returns a dictionary of parameter values for a given ion preset (e.g., '4d_d1', '5d_d5').

    Parameters:
    - preset (str): One of '3d_d1', '4d_d1', '5d_d1', '5d_d5', etc.
    - resolution (int): Number of values per parameter to generate (linspace)

    Returns:
    - Dictionary of param name → list of values
      (also includes 'N' directly for particle number)
    """

    # Define supported presets
    presets = {
        "3d_d1": {
            "N": [1],
            "t": (0.1, 0.3),
            "U": (3.0, 6.0),
            "J": (0.6, 1.2),
            "lbd": (0.02, 0.07),
            "g": (0.1, 0.2),
            "Jphm": (0.1, 0.2),
        },
        "4d_d1": {
            "N": [1],
            "t": (0.3, 0.5),
            "U": (1.5, 3.0),
            "J": (0.4, 0.6),
            "lbd": (0.1, 0.2),
            "g": (0.02, 0.1),
            "Jphm": (0.02, 0.1),
        },
        "5d_d1": {
            "N": [1],
            "t": (0.6, 1.0),
            "U": (1.0, 3.0),
            "J": (0.2, 0.5),
            "lbd": (0.2, 0.4),
            "g": (0.0, 0.02),
            "Jphm": (0.0, 0.02),
        }
    }

    # Error if unknown preset
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(
            f"Preset '{preset}' not defined.\n"
            f"Available presets: {available}"
        )

    selected = presets[preset]

    # Interpolate ranges using linspace (only for tuple entries)
    values = {
        k: np.linspace(v[0], v[1], resolution) if isinstance(v, tuple) else v
        for k, v in selected.items()
    }

    return values
