import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config.grid_builder import generate_lat_grid
from t2g_jt_soc.lat_simulation.lat_runner import run_all_simulations


PRESET = "5d_d1_r1"
RESOLUTION = 4

if __name__ == "__main__":
    grid = generate_lat_grid(PRESET, RESOLUTION)
    print(grid)
    run_all_simulations(grid, parallel=False)
