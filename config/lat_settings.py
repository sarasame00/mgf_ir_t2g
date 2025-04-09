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

LAT_OUTPUT_DIR = "t2g_jt_soc/lat_data/lat_results"  # Save path
LAT_CSV_DIR = "t2g_jt_soc/lat_data/simulated_values_lat.csv"