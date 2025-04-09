import h5py

def load_correl_data(h5_path):
    """
    Loads all data needed to plot orbital, spin-orbital, and spin-exchange correlations
    from a single .hdf5 simulation output file.

    Parameters:
    - h5_path (str): Path to the .hdf5 file.

    Returns:
    - dict: Dictionary containing irrBZ, k_sz, and all correlators.
    """
    with h5py.File(h5_path, "r") as f:
        return {
            "irrBZ": f["irrBZ"][:],
            "k_sz": f["k_sz"][()],
            "corrdiag": f["correldiag"][:],
            "corroffd": f["correloffd"][:],
            "filename": h5_path.split("/")[-1].replace(".hdf5", "")
        }
