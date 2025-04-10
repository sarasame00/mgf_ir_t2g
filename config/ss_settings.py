SS_CONSTANTS = {
    "qmax": 1.2, # Grid range [-QMAX, QMAX]
    "size_grid": 101,
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SS_CSV_DIR = BASE_DIR / "single_site" / "ss_data" / "simulated_values_ss.csv"
SS_OUTPUT_DIR = BASE_DIR / "single_site" / "ss_data" / "ss_results"


SS_CSV_HEADER = [
    'N', 'U', 'J', 'g', 'lbd', 'B', 'qmax', 'size_grid', 'timestamp'
]