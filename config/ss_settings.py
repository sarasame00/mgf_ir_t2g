SS_CONSTANTS = {
    "qmax": 1.2, # Grid range [-QMAX, QMAX]
    "size_grid": 101,
}

SS_CSV_HEADER = [
    'N', 'U', 'J', 'g', 'lbd', 'B', 'qmax', 'size_grid', 'timestamp'
]

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SS_CSV_NAME = "simulated_values_ss.csv"
SS_CSV_DIR = BASE_DIR / "single_site" / "ss_data" / SS_CSV_NAME
SS_OUTPUT_DIR = BASE_DIR / "single_site" / "ss_data" / "ss_results"

SS_GD_ID_DIR = '1lfVuc2xSYqTm2Xn4cjzY_wDl-EnN3yZW' #ID of the direcoctory to save the files in GDrive

