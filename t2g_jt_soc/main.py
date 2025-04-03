from simulation.runner import run_all_simulations
from config.parameters import generate_grid
import os

if __name__ == "__main__":
    grid = generate_grid()
    base_dir = os.path.dirname(__file__)  # Gets the folder where main.py lives
    csv_path = os.path.join(base_dir, "data", "simulated_values.csv")
    data_dir = os.path.join(base_dir, "data", "results")


    run_all_simulations(grid, csv_path, data_dir)
