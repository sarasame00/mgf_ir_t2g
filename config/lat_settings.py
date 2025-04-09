LAT_CONSTANTS = {
    "T": [10, 4],
    "wmax": 8,
    "w0": 0.1,
    "k_sz": 24,
    "diis_mem": 5
}

LAT_CSV_HEADER = [
    'T', 'wm', 'N', 't', 'U', 'J', 'Jphm', 'w0', 'g', 'lbd',
    'k_sz', 'diis_mem', 'timestamp'
]

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LAT_CSV_DIR = BASE_DIR / "t2g_jt_soc" / "lat_data" / "simulated_values_lat.csv"
LAT_OUTPUT_DIR = BASE_DIR / "t2g_jt_soc" / "lat_data" / "lat_results"

