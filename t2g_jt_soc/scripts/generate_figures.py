import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysis.visualize import plot_single_correlation
from analysis.loaders import load_figure3_data
# Path to your simulation HDF5 file (update this as needed!)
h5_path = "t2g_jt_soc/data/results/20250403131701373461_47523.hdf5"

data = load_figure3_data(h5_path)
print(data['diag'])

x = np.arange(0, len(data['diag']))


q_norm = np.linalg.norm(data['qpts'], axis=1)  # simple high-symmetry approximation
sort_idx = np.argsort(q_norm)

q_sorted = q_norm[sort_idx]
cov_sorted = (data['diag'] + data['offd'] + data['crs1'] + data['crs2'])[sort_idx]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(q_sorted, cov_sorted, lw=2, marker='o')
ax.set_xlabel("q (arbitrary units)")
ax.set_ylabel("Orbital–orbital covariance")
ax.set_title("Figure 3b: Covariance vs q in iBZ")
ax.grid(True)

plt.savefig("t2g_jt_soc/data/figures", dpi=300, bbox_inches="tight")
'''
# Output figure folder
fig_dir = "t2g_jt_soc/data/figures"
os.makedirs(fig_dir, exist_ok=True)

# Generate all 4 subplots separately
plot_single_correlation(h5_path, 'correldiag', r"$\langle n^\alpha_q n^\alpha_{-q} \rangle$", save_path=os.path.join(fig_dir, "fig3a_correldiag.png"))
plot_single_correlation(h5_path, 'correloffd', r"$\langle n^\alpha_q n^\beta_{-q} \rangle$", save_path=os.path.join(fig_dir, "fig3b_correloffd.png"))
plot_single_correlation(h5_path, 'correlcrs1', r"$\langle \tau^x_q \tau^x_{-q} \rangle$", save_path=os.path.join(fig_dir, "fig3c_crs1.png"))
plot_single_correlation(h5_path, 'correlcrs2', r"$\langle \tau^y_q \tau^y_{-q} \rangle$", save_path=os.path.join(fig_dir, "fig3d_crs2.png"))
'''