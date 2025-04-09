SS_CONSTANTS = {
    "qmax": 1.2, # Grid range [-QMAX, QMAX]
    "size_grid": 101,
    "B": 0.1
}

SS_OUTPUT_DIR = "single_site/ss_data/ss_results"  # Save path
SS_CSV_DIR = "single_site/ss_data/simulated_values_ss.csv"
SS_CSV_HEADER = [
    'N', 'U', 'J', 'g', 'lbd', 'B', 'qmax', 'size_grid', 'timestamp'
]