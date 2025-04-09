import matplotlib.pyplot as plt
import os
from .utils import take_borders, antifourier

outdir = "t2g_jt_soc/data/figures"

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["ytick.labelsize"] = 13


def plot_orbital_momentum(data, outdir=outdir):
    """
    Plot the orbital-orbital correlation in momentum space.

    Parameters:
    - data (dict): Output from `load_correl_data`, containing correlation values and metadata.
    - outdir (str): Path to save the generated figure.
    """
    # Compute the orbital charge correlator
    orbcharge = 4 * (data["corrdiag"] - data["corroffd"])
    dist, orbcharge_k = take_borders(data["irrBZ"], orbcharge)
    k_sz = data["k_sz"]
    # Create the figure and axis explicitly
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dist, orbcharge_k)

    # Set axis labels and grid
    ax.set_ylabel(r"$\langle \delta \mathbb{T}_\mathbf{q} \delta \mathbb{T}_\mathbf{q} \rangle$")
    ax.set_xlabel(r"$x_j - x_i$")

    kymin, kymax = ax.get_ylim()
    ax.set_ylim(kymin, kymax)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlim(dist[0], dist[-1])
    ax.set_xticks([dist[i * (k_sz // 2)] for i in range(8)])
    kymin, kymax = ax.get_ylim()
    ax.vlines([dist[i * 12] for i in range(8)],kymin, kymax, color='k', lw=0.5)

    ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R', 'X', 'M', 'R'])

    # Save the plot
    os.makedirs(outdir, exist_ok=True)
    fname = f"{data['filename']}_orbital_momentum.png"
    fig.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close(fig)

def plot_orbital_real(data, outdir=outdir):
    """
    Plot the orbital-orbital correlation in real space.

    Parameters:
    - data (dict): Output from `load_correl_data`, containing correlation values and metadata.
    - outdir (str): Directory to save the output figure.
    """
    # Compute orbital correlation in real space
    orbcharge = 4 * (data["corrdiag"] - data["corroffd"])
    x, orbcharge_r = antifourier(data["k_sz"], data["irrBZ"], orbcharge)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, orbcharge_r, marker='o')

    # Axis configuration
    ax.set_ylabel(r"$\langle \delta \mathbb{T}_i \delta \mathbb{T}_j \rangle$")
    ax.set_xlabel(r"$x_j - x_i$")
    ax.set_ylim(-0.18, 0.18)
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
    

    # Save to file
    os.makedirs(outdir, exist_ok=True)
    fname = f"{data['filename']}_orbital_real.png"
    fig.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close(fig)

def plot_spin_momentum(data, outdir=outdir):
    """
    Plot spin-orbital correlation in momentum space.
    """
    spincharge = 2 * (3 * data["corrdiag"] + data["corroffd"])
    dist, spin_k = take_borders(data["irrBZ"], spincharge)
    k_sz = data["k_sz"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dist, spin_k)

    ax.set_ylabel(r"$\langle \delta \mathbb{W}_\mathbf{q} \delta \mathbb{W}_\mathbf{q} \rangle$")
    ax.set_xlabel(r"$\mathbf{q}$ along iBZ path")
    
    kymin, kymax = ax.get_ylim()
    ax.set_ylim(kymin, kymax)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlim(dist[0], dist[-1])
    ax.set_xticks([dist[i * (k_sz // 2)] for i in range(8)])
    kymin, kymax = ax.get_ylim()
    ax.vlines([dist[i * 12] for i in range(8)],kymin, kymax, color='k', lw=0.5)

    ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R', 'X', 'M', 'R'])

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{data['filename']}_spin_momentum.png", dpi=300)
    plt.close(fig)


def plot_spin_real(data, outdir=outdir):
    """
    Plot spin-orbital correlation in real space.
    """
    spincharge = 2 * (3 * data["corrdiag"] + data["corroffd"])
    x, spin_r = antifourier(data["k_sz"], data["irrBZ"], spincharge)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, spin_r, marker='o')

    ax.set_ylabel(r"$\langle \delta \mathbb{W}_i \delta \mathbb{W}_j \rangle$")
    ax.set_xlabel(r"$x_j - x_i$")
    ax.set_ylim(-0.18, 0.18)
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{data['filename']}_spin_real.png", dpi=300)
    plt.close(fig)

def plot_spinexchange_momentum(data, outdir=outdir):
    """
    Plot spin-exchange correlation in momentum space.
    """
    spinexchange = 4 * data["corroffd"]
    dist, exchange_k = take_borders(data["irrBZ"], spinexchange)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dist, exchange_k)

    ax.set_ylabel(r"$\langle \delta n_{\uparrow\mathbf{q}} \delta n_{\downarrow\mathbf{q}} \rangle$")
    ax.set_xlabel(r"$\mathbf{q}$ along iBZ path")
    ax.grid(True)

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{data['filename']}_spinexchange_momentum.png", dpi=300)
    plt.close(fig)
