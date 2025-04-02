from simulation.runner import run_all_simulations
from config.parameters import generate_grid

if __name__ == "__main__":
    grid = generate_grid()
    csv_path = "data/simulated_values.csv"
    data_dir = "data/results"

    run_all_simulations(grid, csv_path, data_dir)
