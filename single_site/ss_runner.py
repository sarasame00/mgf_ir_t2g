import os
import numpy as np
import csv
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Lock

from single_site.ss_simulation import eigobj
from config.ss_settings import SS_OUTPUT_DIR, SS_CSV_DIR, SS_GD_ID_DIR, SS_CSV_HEADER, SS_CSV_NAME
from drive_utils import upload_file_to_drive, get_completed_params_from_drive, update_and_upload_csv, download_csv_from_drive

# Locks for thread-safe operations
print_lock = Lock()
csv_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def run_ss_simulation(param_tuple, already_done=None, upload_to_drive=True):
    if already_done is None:
        already_done = []

    N, U, J, g, lbd, B, Qmax, size_grid = param_tuple

    safe_print(f"\nRunning simulation for N={N}, U={U:.2f}, J={J:.2f}, g={g:.3f}, B={B:.3f}, lbd={lbd:.3f}")

    rounded_param = tuple(round(x, 3) for x in param_tuple)

    if rounded_param in already_done:
        safe_print(f"Skipping completed simulation: {rounded_param}")
        return

    now = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    filepath = os.path.join(SS_OUTPUT_DIR, str(now))
    os.makedirs(SS_OUTPUT_DIR, exist_ok=True)

    Qx, Qy = np.meshgrid(np.linspace(-Qmax, Qmax, size_grid), np.linspace(-Qmax, Qmax, size_grid))
    emap = np.zeros((size_grid, size_grid))
    for i in range(size_grid):
        for j in range(size_grid):
            emap[i, j] = eigobj([Qx[i, j], Qy[i, j]], U, J, lbd, N, B, g)
    emap -= np.min(emap)

    np.savetxt(filepath, np.array((Qx, Qy, emap)).reshape((3, emap.size)).T)

    if upload_to_drive and os.path.exists(filepath):
        upload_file_to_drive(filepath=filepath, filename=str(now), parent_id=SS_GD_ID_DIR, overwrite=True)

    new_row = list(param_tuple) + [int(now)]

    with csv_lock:
        update_and_upload_csv(new_row, SS_CSV_DIR, SS_GD_ID_DIR, SS_CSV_NAME, SS_CSV_HEADER)

    safe_print(f"✅ Saved result to {filepath} and updated CSV.")


def run_all_ss_simulations(param_tuples, n_jobs=-1, parallel=True, upload_to_drive=True):
    os.makedirs(SS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SS_CSV_DIR), exist_ok=True)

    if not os.path.exists(SS_CSV_DIR) or os.path.getsize(SS_CSV_DIR) == 0:
        with open(SS_CSV_DIR, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SS_CSV_HEADER)

    already_done = get_completed_params_from_drive("simulated_values_ss.csv", SS_GD_ID_DIR)

    try:
        if parallel:
            Parallel(n_jobs=n_jobs)(
                delayed(run_ss_simulation)(val, already_done, upload_to_drive)
                for val in tqdm(param_tuples, desc="Running simulations (parallel)")
            )
        else:
            for val in tqdm(param_tuples, desc="Running simulations (serial)"):
                run_ss_simulation(val, already_done, upload_to_drive)

    except KeyboardInterrupt:
        safe_print("\n❌ Simulation interrupted by user.")

    safe_print("\n✅ All simulations completed.")
