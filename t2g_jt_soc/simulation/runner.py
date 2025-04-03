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
    Runs a single DysonSolver simulation and appends result to CSV.
    Thread-safe for joblib multiprocessing.
    """

    # Load previously completed simulations
    already = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for line in reader:
                already.add(tuple([float(s) for s in line[:-1]]))

    if val in already:
        safe_print(f"⏭️  Skipping: already done → {val}")
        return

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = os.path.join(data_dir, now)

    # Log start of the simulation
    safe_print(f"🚀 Starting → T={val[0]} N={val[2]} U={val[4]} g={val[8]} lbd={val[9]}")

    # Run DysonSolver
    solver = DysonSolver(*val, fl=out_path + ".out")
    solver.solve(diis_active=True, tol=5e-6)
    solver.save(out_path)

    safe_print(f"✅ Finished → saved {now}.out")

    # Append result to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(val) + [int(now)])


def run_all_simulations(parameter_grid, csv_path, data_dir, n_jobs=-1):
    """
    Run all Dyson simulations in parallel (joblib + tqdm).
    Gracefully handles Ctrl+C interruption.
    """
    # Ensure folders exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Initialize CSV file with header if needed
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    try:
        # Launch all jobs with progress bar
        Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(val, csv_path, data_dir)
            for val in tqdm(parameter_grid, desc="Running simulations")
        )
    except KeyboardInterrupt:
        safe_print("\n❌ Simulation interrupted by user (Ctrl+C). Exiting gracefully...")
