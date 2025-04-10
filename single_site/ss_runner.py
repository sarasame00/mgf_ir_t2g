import os
import numpy as np
import csv
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Lock

from single_site.ss_simulation import eigobj
from config.ss_settings import SS_OUTPUT_DIR, SS_CSV_DIR, SS_CSV_HEADER

# Lock to ensure thread-safe printing when running in parallel
print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def run_ss_simulation(param_tuple):
    """
    Run a single-site JT+SOC simulation and save the result.

    Parameters:
    - param_tuple: (N, U, J, g, lbd, B, Qmax, size_grid)

    Returns:
    - emap (np.ndarray): Normalized energy map for the given parameter set
    """
    N, U, J, g, lbd, B, Qmax, size_grid = param_tuple

    safe_print(f"\nRunning simulation for N={N}, U={U:.2f}, J={J:.2f}, g={g:.3f}, B={B:.3f}, lbd={lbd:.3f}")

    # Check if this parameter set has already been completed
    already = set()
    if os.path.exists(SS_CSV_DIR):
        with open(SS_CSV_DIR, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for line in reader:
                already.add(tuple([float(s) for s in line[:-1]]))

    if param_tuple in already:
        safe_print(f"Skipping completed simulation: {param_tuple}")
        return

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

    # Log the parameter set and timestamp to the CSV file
    with open(SS_CSV_DIR, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(param_tuple) + [int(now)])

    safe_print(f"Saved result to {filepath}")
    return emap

def run_all_ss_simulations(param_tuples, n_jobs=-1, parallel=True):
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
                delayed(run_ss_simulation)(param_tuple)
                for param_tuple in tqdm(param_tuples, desc="Running simulations (parallel)")
            )
        else:
            for param_tuple in tqdm(param_tuples, desc="Running simulations (serial)"):
                run_ss_simulation(param_tuple)

    except KeyboardInterrupt:
        safe_print("\n❌Simulation interrupted by user.")
    
    safe_print("\n✅ All simulations completed.")
