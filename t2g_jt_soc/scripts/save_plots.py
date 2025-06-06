import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

print(ROOT_DIR)
from analysis.loaders import load_correl_data
from analysis.visualize import plot_orbital_momentum, plot_spin_momentum, plot_orbital_real,plot_spin_real,plot_spinexchange_momentum

h5_paths = ['2025041713055771033425185',
 '2025041713215049263925185',
 '2025041713384777077225185',
 '2025041713491499712525185',
 '2025041714384073368326514',
 '2025041714473582481026514',
 '2025041714563439649226514',
 '2025041715050670570526514',
 '2025041716030744710426973',
 '2025041716094655060426973']

outdir= str(ROOT_DIR) + '/lat_data/lat_figures'

for i in h5_paths:
    path = str(ROOT_DIR) + '/lat_data/lat_results/' + i + '.hdf5'
    data = load_correl_data(path)
    plot_orbital_momentum(data, outdir)
    plot_orbital_real(data, outdir)
    plot_spin_momentum(data, outdir)
    plot_spin_real(data , outdir)
    plot_spinexchange_momentum(data, outdir)