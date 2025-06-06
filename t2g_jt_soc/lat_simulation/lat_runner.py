import os
import csv
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Lock

from lat_simulation.dyson_solver import DysonSolver
from config.lat_settings import LAT_CSV_HEADER, LAT_CSV_DIR, LAT_OUTPUT_DIR, LAT_GD_ID_DIR, LAT_CSV_NAME

from drive_utils import update_and_upload_csv, get_completed_params_from_drive

# Global print lock to avoid overlap in logs
print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def run_single_simulation(val, upload_to_drive=True):
    """
    Run a single DysonSolver simulation and record its result.

    This function:
    - Checks if the parameter set `val` has already been simulated.
    - Runs the Dyson equation solver for that parameter set.
    - Saves the results to disk (both .out and .hdf5 files).
    - Appends a summary row to the tracking CSV.

    Parameters:
    - val (tuple): A single set of parameters(T, wmax, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)

    Note:
    - Compatible with joblib multiprocessing (thread-safe).
    """

    # Round the param tuple before comparison (3 decimal places)
    rounded_param = tuple(round(x, 3) for x in val)
    if upload_to_drive:
        already = get_completed_params_from_drive("simulated_values_lat.csv", LAT_GD_ID_DIR)
    else:
        already = []

    if rounded_param in already:
        safe_print(f"Skipping completed simulation: {rounded_param}")
        return
    else:
        safe_print(f"Running new simulation: {rounded_param}")

    # Create a unique timestamp + PID-based identifier
    now = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}{os.getpid()}"
    out_path = os.path.join(LAT_OUTPUT_DIR, now)

    # Run the solver for this parameter set
    solver = DysonSolver(*val, fl=out_path + ".out")
    solver.solve(diis_active=True, tol=5e-6)

    solver.save(out_path, upload_to_drive) # Save hdf5
    
    # Upload csv
    new_row = list(val) + [int(now)]
    if upload_to_drive:
        update_and_upload_csv(new_row, LAT_CSV_DIR, LAT_GD_ID_DIR, LAT_CSV_NAME, LAT_CSV_HEADER)
    else:
        df = pd.read_csv(LAT_CSV_DIR)
        # Detect if it's a single row (list of values) or list of rows
        if isinstance(new_row[0], (int, float, str)):
            # It's a single row → wrap it
            new_row = [new_row]

        # Create DataFrame
        new_df = pd.DataFrame(new_row, columns=LAT_CSV_HEADER)

        # Concatenate safely
        combined_df = pd.concat([df, new_df], ignore_index=True).drop_duplicates()

        # Save locally
        combined_df.to_csv(LAT_CSV_DIR, index=False)
    
    # Log the simulation finish
    safe_print(f"✅ Finished → saved {now}")

def run_all_simulations(parameter_grid, n_jobs=-1, parallel=True, upload_to_drive=True):
    """
    Run DysonSolver simulations either in parallel or serially.

    Parameters:
    - parameter_grid (list of tuples): All parameter sets to simulate.
    - n_jobs (int): Number of workers (only used if parallel=True).
    - parallel (bool): Whether to run in parallel or not.
    """

    # Ensure directories exist
    os.makedirs(LAT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LAT_CSV_DIR), exist_ok=True)

    # Initialize CSV log with header if not present
    if not os.path.exists(LAT_CSV_DIR) or os.path.getsize(LAT_CSV_DIR) == 0:
        with open(LAT_CSV_DIR, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(LAT_CSV_HEADER)

    try:
        if parallel:
            # Run in parallel using joblib
            Parallel(n_jobs=n_jobs)(
                delayed(run_single_simulation)(val, upload_to_drive)
                for val in tqdm(parameter_grid, desc="Running simulations (parallel)")
            )
        else:
            # Run sequentially for debugging or easier logging
            for val in tqdm(parameter_grid, desc="Running simulations (serial)"):
                run_single_simulation(val, upload_to_drive)

    except KeyboardInterrupt:
        safe_print("\n❌ Simulation interrupted by user.")

    safe_print("\n✅ All simulations completed.")
