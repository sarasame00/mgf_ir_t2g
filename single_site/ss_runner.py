import os
import numpy as np
import csv
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Lock

from single_site.ss_simulation import eigobj
from config.ss_settings import SS_OUTPUT_DIR, SS_CSV_DIR, SS_GD_ID_DIR, SS_CSV_HEADER, SS_CSV_NAME
from drive_utils import upload_file_to_drive, get_completed_params_from_drive, update_and_upload_csv

# Global locks for thread-safe logging and CSV updates
print_lock = Lock()
csv_lock = Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function for cleaner logs in parallel jobs."""
    with print_lock:
        print(*args, **kwargs)

def run_ss_simulation(param_tuple, upload_to_drive=True):
    """
    Run a single site simulation for a given parameter set.

    This function:
    - Checks whether the given parameter set has already been simulated.
    - Computes the energy map from the eigobj solver over a 2D Q-grid.
    - Saves the energy map to disk and optionally uploads it to Google Drive.
    - Appends a summary of the simulation to the tracking CSV.

    Parameters:
    - param_tuple (tuple): Parameters for simulation (N, U, J, g, lbd, B, Qmax, size_grid)
    - upload_to_drive (bool): Whether to upload results and CSV to Google Drive
    """

    N, U, J, g, lbd, B, Qmax, size_grid = param_tuple

    safe_print(f"\nRunning simulation for N={N}, U={U:.2f}, J={J:.2f}, g={g:.3f}, B={B:.3f}, lbd={lbd:.3f}")

    # Check if this parameter set has already been simulated
    rounded_param = tuple(round(x, 3) for x in param_tuple)
    if upload_to_drive:
        already = get_completed_params_from_drive("simulated_values_lat.csv", SS_GD_ID_DIR)
    else:
        already = []

    if rounded_param in already:
        safe_print(f"Skipping completed simulation: {rounded_param}")
        return

    # Unique filename using timestamp
    now = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    filepath = os.path.join(SS_OUTPUT_DIR, str(now))
    os.makedirs(SS_OUTPUT_DIR, exist_ok=True)

    # Generate 2D Q grid and compute energy map
    Qx, Qy = np.meshgrid(np.linspace(-Qmax, Qmax, size_grid), np.linspace(-Qmax, Qmax, size_grid))
    emap = np.zeros((size_grid, size_grid))
    for i in range(size_grid):
        for j in range(size_grid):
            emap[i, j] = eigobj([Qx[i, j], Qy[i, j]], U, J, lbd, N, B, g)

    # Normalize energy map
    emap -= np.min(emap)

    # Save energy map to file
    np.savetxt(filepath, np.array((Qx, Qy, emap)).reshape((3, emap.size)).T)

    # Upload result file to Google Drive
    if upload_to_drive and os.path.exists(filepath):
        upload_file_to_drive(filepath=filepath, filename=str(now), parent_id=SS_GD_ID_DIR, overwrite=True)

    # Create new CSV row for this result
    new_row = list(param_tuple) + [int(now)]

    # Update CSV (thread-safe)
    with csv_lock:
        if upload_to_drive:
            update_and_upload_csv(new_row, SS_CSV_DIR, SS_GD_ID_DIR, SS_CSV_NAME, SS_CSV_HEADER)
        else:
            df = pd.read_csv(SS_CSV_DIR)

            # Detect if it's a single row or list of rows
            if isinstance(new_row[0], (int, float, str)):
                new_row = [new_row]

            # Create DataFrame and append
            new_df = pd.DataFrame(new_row, columns=SS_CSV_HEADER)
            combined_df = pd.concat([df, new_df], ignore_index=True).drop_duplicates()

            # Save updated CSV locally
            combined_df.to_csv(SS_CSV_DIR, index=False)

    safe_print(f"✅ Saved result to {filepath} and updated CSV.")

def run_all_ss_simulations(param_tuples, n_jobs=-1, parallel=True, upload_to_drive=True):
    """
    Run all single-site simulations either in parallel or serially.

    Parameters:
    - param_tuples (list of tuples): All parameter sets to simulate.
    - n_jobs (int): Number of workers (only used if parallel=True).
    - parallel (bool): Whether to run in parallel or not.
    - upload_to_drive (bool): Whether to upload results to Google Drive
    """

    # Ensure necessary directories exist
    os.makedirs(SS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SS_CSV_DIR), exist_ok=True)

    # Create CSV file with header if it doesn't exist yet
    if not os.path.exists(SS_CSV_DIR) or os.path.getsize(SS_CSV_DIR) == 0:
        with open(SS_CSV_DIR, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SS_CSV_HEADER)

    try:
        if parallel:
            # Run simulations in parallel using joblib
            Parallel(n_jobs=n_jobs)(
                delayed(run_ss_simulation)(val, upload_to_drive)
                for val in tqdm(param_tuples, desc="Running simulations (parallel)")
            )
        else:
            # Run sequentially for debugging or easier logging
            for val in tqdm(param_tuples, desc="Running simulations (serial)"):
                run_ss_simulation(val, upload_to_drive)

    except KeyboardInterrupt:
        safe_print("\n❌ Simulation interrupted by user.")

    safe_print("\n✅ All simulations completed.")
