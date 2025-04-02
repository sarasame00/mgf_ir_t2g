import os
from datetime import datetime
import csv

from simulation.dyson_solver import DysonSolver  # Dyson equation solver class

def run_all_simulations(parameter_grid, csv_path, data_dir):
    """
    Runs a batch of simulations using the DysonSolver across a predefined parameter grid.
    Only runs simulations that have not already been recorded in the CSV file.
    
    Parameters:
    - parameter_grid (list of tuples): Each tuple defines a full set of model parameters.
    - csv_path (str): Path to the CSV file that logs completed simulations.
    - data_dir (str): Directory to store simulation outputs (.out and .hdf5).
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load previously completed simulations from the CSV log
    already = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # skip header if present
            except StopIteration:
                pass  # file is empty, nothing to skip
            for line in reader:
                already.add(tuple([float(s) for s in line[:-1]]))

    # Open the CSV file in append mode to add new entries
    with open(csv_path, 'a', newline='') as csvfl:
        writer = csv.writer(csvfl)

        # Loop over all parameter sets in the grid
        for val in parameter_grid:
            if val in already:
                # Skip this set if it was already completed
                continue

            # Generate a unique timestamp-based identifier
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            out_path = os.path.join(data_dir, now)

            # Initialize and run the DysonSolver
            solver = DysonSolver(*val, fl=out_path + ".out")
            solver.solve(diis_active=True, tol=5e-6)
            solver.save(out_path)

            # Log the parameters and timestamp to the CSV file
            writer.writerow(list(val) + [int(now)])
