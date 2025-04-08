import os
import csv
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Lock

from simulation.dyson_solver import DysonSolver

# CSV column header
CSV_HEADER = [
    'T', 'wm', 'N', 't', 'U', 'J', 'Jphm', 'w0', 'g', 'lbd',
    'k_sz', 'diis_mem', 'timestamp'
]

# Global print lock to avoid overlap in logs
print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def run_single_simulation(val, csv_path, data_dir):
    """
    Run a single DysonSolver simulation and record its result.

    This function:
    - Checks if the parameter set `val` has already been simulated.
    - Runs the Dyson equation solver for that parameter set.
    - Saves the results to disk (both .out and .hdf5 files).
    - Appends a summary row to the tracking CSV.

    Parameters:
    - val (tuple): A single set of parameters (T, wM, N, t, U, J, ..., etc).
    - csv_path (str): Path to the CSV file tracking completed runs.
    - data_dir (str): Directory where simulation outputs will be saved.

    Note:
    - Compatible with joblib multiprocessing (thread-safe).
    """

    # Load previously completed runs into a set
    already = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for line in reader:
                already.add(tuple([float(s) for s in line[:-1]]))

    # Skip this parameter set if already done
    if val in already:
        safe_print(f"⏭️  Skipping: already done → {val}")
        return

    # Create a unique timestamp + PID-based identifier
    now = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{os.getpid()}"
    out_path = os.path.join(data_dir, now)

    # Log the simulation start (clean print)
    safe_print(f"🚀 Starting → T={val[0]} N={val[2]} U={val[4]} g={val[8]} lbd={val[9]}")

    # Run the solver for this parameter set
    solver = DysonSolver(*val, fl=out_path + ".out")
    solver.solve(diis_active=True, tol=5e-6)
    solver.save(out_path)

    # Log the simulation finish
    safe_print(f"✅ Finished → saved {now}.out")

    # Append the result row to the CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(val) + [int(now)])


def run_all_simulations(parameter_grid, csv_path, data_dir, n_jobs=-1, parallel=True):
    """
    Run DysonSolver simulations either in parallel or serially.

    Parameters:
    - parameter_grid (list of tuples): All parameter sets to simulate.
    - csv_path (str): Path to the CSV tracking file.
    - data_dir (str): Directory where .out and .hdf5 files are stored.
    - n_jobs (int): Number of workers (only used if parallel=True).
    - parallel (bool): Whether to run in parallel or not.
    """

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Initialize CSV log with header if not present
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    try:
        if parallel:
            # Run in parallel using joblib
            Parallel(n_jobs=n_jobs)(
                delayed(run_single_simulation)(val, csv_path, data_dir)
                for val in tqdm(parameter_grid, desc="Running simulations (parallel)")
            )
        else:
            # Run sequentially for debugging or easier logging
            for val in tqdm(parameter_grid, desc="Running simulations (serial)"):
                run_single_simulation(val, csv_path, data_dir)

    except KeyboardInterrupt:
        safe_print("\n❌ Simulation interrupted by user (Ctrl+C).")

