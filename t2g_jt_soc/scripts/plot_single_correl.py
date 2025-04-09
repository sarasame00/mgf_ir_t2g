import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysis.loaders import load_correl_data
from analysis.visualize import (
    plot_orbital_momentum, plot_orbital_real,
    plot_spin_momentum, plot_spin_real,
    plot_spinexchange_momentum
)

h5_path = "t2g_jt_soc/data/results/20250403131701373461_47523.hdf5"  


data = load_correl_data(h5_path)

plot_orbital_momentum(data)
plot_orbital_real(data)
plot_spin_momentum(data)
plot_spin_real(data)
plot_spinexchange_momentum(data)
