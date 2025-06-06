import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config.grid_builder import generate_lat_grid
from t2g_jt_soc.lat_simulation.lat_runner import run_all_simulations


PRESET = "5d_d1_r2"
RESOLUTION = 3

presets = ["3d_d5_r1","4d_d5_r1","5d_d5_r1"]
if __name__ == "__main__":
    for pres in presets:
        grid = generate_lat_grid(pres, RESOLUTION)
        run_all_simulations(grid, parallel=True)
