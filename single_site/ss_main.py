import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config.grid_builder import generate_ss_grid
from single_site.ss_runner import run_all_ss_simulations


PRESET = "3d_d1_r2"         # Choose from: '3d_d1_r1', '4d_d1_r1', '5d_d1_r1', etc.
RESOLUTION = 3           # Parameter resolution (per dimension)

presets = ["3d_d1_r2","4d_d1_r2","5d_d1_r2"]
if __name__ == "__main__":
    for pres in presets:
        param_tuples = generate_ss_grid(pres, RESOLUTION)
        run_all_ss_simulations(param_tuples, parallel=False)
