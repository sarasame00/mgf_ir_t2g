import os
import numpy as np
import csv
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Lock

from single_site.ss_simulation import eigobj
from config.ss_settings import SS_OUTPUT_DIR, SS_CSV_DIR, SS_GD_ID_DIR, SS_CSV_HEADER, SS_CSV_NAME

import os
from drive_utils import upload_file_to_drive, get_completed_params_from_drive, update_and_upload_csv

# Lock to ensure thread-safe printing when running in parallel
print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def run_ss_simulation(param_tuple, upload_to_drive=True):
    """
    Run a single-site JT+SOC simulation and save the result.

    Parameters:
    - param_tuple: (N, U, J, g, lbd, B, Qmax, size_grid)

    Returns:
    - emap (np.ndarray): Normalized energy map for the given parameter set
    """
    N, U, J, g, lbd, B, Qmax, size_grid = param_tuple

    safe_print(f"\nRunning simulation for N={N}, U={U:.2f}, J={J:.2f}, g={g:.3f}, B={B:.3f}, lbd={lbd:.3f}")

    # Round the param tuple before comparison (3 decimal places)
    rounded_param = tuple(round(x, 3) for x in param_tuple)

    already = get_completed_params_from_drive("simulated_values_ss.csv", SS_GD_ID_DIR)

    if rounded_param in already:
        safe_print(f"Skipping completed simulation: {rounded_param}")
        return
    else:
        safe_print(f"Running new simulation: {rounded_param}")


    # Generate a unique timestamp for the output file
    now = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    # Create output directory if it does not exist
    os.makedirs(SS_OUTPUT_DIR, exist_ok=True)

    # Create Qx, Qy grid
    Qx, Qy = np.meshgrid(
        np.linspace(-Qmax, Qmax, size_grid),
        np.linspace(-Qmax, Qmax, size_grid)
    )
    emap = np.zeros((size_grid, size_grid))

    # Compute the lowest eigenvalue at each point on the grid
    for i in range(size_grid):
        for j in range(size_grid):
            emap[i, j] = eigobj([Qx[i, j], Qy[i, j]], U, J, lbd, N, B, g)

    # Normalize the energy map
    emap -= np.min(emap)

    # Save the energy map to a timestamped file
    filepath = os.path.join(SS_OUTPUT_DIR, str(now))
    np.savetxt(filepath, np.array((Qx, Qy, emap)).reshape((3, emap.size)).T)

    if upload_to_drive and os.path.exists(filepath):
        upload_file_to_drive(
            filepath=filepath,
            filename=str(now),
            parent_id=SS_GD_ID_DIR,
            overwrite=True  
        )

    
    # Upload csv
    new_row = list(param_tuple) + [int(now)]
    update_and_upload_csv(new_row, SS_CSV_DIR, SS_GD_ID_DIR, SS_CSV_NAME, SS_CSV_HEADER)
    safe_print(f"Saved result to {filepath}")
    return emap

def run_all_ss_simulations(param_tuples, n_jobs=-1, parallel=True, upload_to_drive=True):
    """
    Run all single-site simulations using the given parameter grid.

    Parameters:
    - param_tuples: List of parameter tuples
    - n_jobs: Number of parallel jobs to use (-1 = use all available cores)
    - parallel: Whether to run in parallel (True) or serially (False)
    """
    # Initialize the CSV file with a header if it does not exist
    if not os.path.exists(SS_CSV_DIR) or os.path.getsize(SS_CSV_DIR) == 0:
        with open(SS_CSV_DIR, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SS_CSV_HEADER)

    try:
        if parallel:
            Parallel(n_jobs=n_jobs)(
                delayed(run_ss_simulation)(param_tuple, upload_to_drive)
                for param_tuple in tqdm(param_tuples, desc="Running simulations (parallel)")
            )
        else:
            for param_tuple in tqdm(param_tuples, desc="Running simulations (serial)"):
                run_ss_simulation(param_tuple, upload_to_drive)

    except KeyboardInterrupt:
        safe_print("\n❌Simulation interrupted by user.")
    
    safe_print("\n✅ All simulations completed.")
