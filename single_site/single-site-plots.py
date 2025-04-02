import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# === Custom colormap ===
colors = [
    (0.0, "black"),
    (0.15, "yellow"),
    (0.25, "orange"),
    (0.35, "red"),
    (0.5, "magenta"),
    (0.65, "blue"),
    (0.85, "cyan"),
    (1.0, "white")
]
custom_cmap = LinearSegmentedColormap.from_list("custom_map", colors)

# === Parameters ===
N_values = range(1, 6)
lb_values = [0, 0.1, 0.3, 0.5]
shape = (101, 101)

data_folder = "PaperFigs/T0data/"
plot_folder = "PaperFigs/T0plots/"
grid_plot_path = "PaperFigs/T0plots/combined_energy_map.png"

os.makedirs(plot_folder, exist_ok=True)

# === Preload all data to compute global vmin/vmax ===
emap_dict = {}
for lb in lb_values:
    soc = int(lb * 10)
    for N in N_values:
        filename = f"{N}N_{soc}SOC_lowee.txt"
        filepath = os.path.join(data_folder, filename)
        if os.path.isfile(filepath):
            data = np.loadtxt(filepath)
            Qx = data[:, 0].reshape(shape)
            Qy = data[:, 1].reshape(shape)
            emap = data[:, 2].reshape(shape)
            emap_dict[(N, lb)] = (Qx, Qy, emap)

# === Global vmin/vmax with 15% range to enhance contrast ===
all_emaps = [v[2] for v in emap_dict.values()]
global_vmin = min(map(np.min, all_emaps))
global_vmax = global_vmin + (max(map(np.max, all_emaps)) - global_vmin) * 0.15

# === Create subplot grid ===
fig, axes = plt.subplots(
    nrows=len(lb_values),
    ncols=len(N_values),
    figsize=(3 * len(N_values), 3 * len(lb_values)),
    sharex=True,
    sharey=True
)
axes = np.array(axes)

# === Main loop: plot individual and add to grid ===
for i, lb in enumerate(lb_values):
    soc = int(lb * 10)
    for j, N in enumerate(N_values):

        filename = f"{N}N_{soc}SOC_lowee.png"

        print('→ Plotting: '+ filename)

        ax = axes[i, j]
        key = (N, lb)

        if key not in emap_dict:
            ax.axis("off")
            continue

        Qx, Qy, emap = emap_dict[key]

        # === Individual plot ===
        fig_ind, ax_ind = plt.subplots()
        c_ind = ax_ind.pcolormesh(Qx, Qy, emap, cmap=custom_cmap, shading='gouraud',
                                  vmin=global_vmin, vmax=global_vmax)
        fig_ind.colorbar(c_ind, ax=ax_ind)
        ax_ind.set_title(f"N={N}, ξ={lb}eV")
        ax_ind.set_xlabel("Qx")
        ax_ind.set_ylabel("Qy")

        plt.savefig(os.path.join(plot_folder, filename), dpi=300, bbox_inches='tight')
        plt.close(fig_ind)

        # === Add to grid plot ===
        c_grid = ax.pcolormesh(Qx, Qy, emap, cmap=custom_cmap, shading='gouraud',
                               vmin=global_vmin, vmax=global_vmax)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_aspect('equal')

        if i == len(lb_values) - 1:
            ax.set_xlabel(f"N={N}", fontsize=12)
        else:
            ax.set_xticklabels([])

        if j == 0:
            ax.set_ylabel(f"$\\xi$={lb}eV", fontsize=12)
        else:
            ax.set_yticklabels([])

print('→ Plotting: combined_energy_map.png')

# === Shared colorbar ===
fig.subplots_adjust(right=0.88, wspace=0.1, hspace=0.1)

# Define position for colorbar [left, bottom, width, height]
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # right side
fig.colorbar(c_grid, cax=cbar_ax, label="Energy")

# Save and show
plt.savefig(grid_plot_path, dpi=300)

print("\n ✅ All plots saved.")
