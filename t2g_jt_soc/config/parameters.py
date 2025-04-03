def generate_grid():
    """
    Returns a list of tuples, each containing a parameter set:
    (T, wm, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)

    Each parameter is defined as follows:

    - T         : Temperature (Kelvin). Used to define β = 1/kT.
    - wm        : Cutoff frequency for the IR basis (in eV), usually denoted Λ.
    - N         : Target total particle number (per site or unit cell).
    - t         : Hopping amplitude for the tight-binding dispersion.
    - U         : On-site Coulomb repulsion (Hubbard interaction).
    - J         : Hund's exchange coupling (favoring spin/orbital alignment).
    - Jphm      : Orbital exchange interaction mediated by phonons (J_phm).
    - w0        : Bare phonon frequency ω₀ (Einstein phonon model).
    - g         : Electron-phonon Jahn-Teller coupling strength.
    - lbd       : Spin-orbit coupling constant λ (from H_SOC = λ L · S).
    - k_sz      : Momentum grid size in each direction (k-point mesh is k_sz³).
    - diis_mem  : Number of DIIS vectors used for convergence acceleration.
    """

    # Sweep over: filling (N), hopping (t), interaction parameters, and temperature (T)
    Nls = [1, 2, 3, 4, 5]                    # Electron number / filling levels
    tls = [0.05, 0.2, 1.2]                   # Hopping amplitudes

    # (U, J, lbd): Electron-electron interactions and spin-orbit coupling
    Usocls = [
        (4, 0.8, 0.05),     # Strong U, strong J, weak SOC
        (2.5, 0.2, 0.3),    # Moderate U/J, strong SOC
        (0.5, 0.04, 0.3),   # Weak U/J, strong SOC
        (4, 0.8, 0),        # Strong U/J, no SOC
        (2.5, 0.2, 0),      # Intermediate U/J, no SOC
        (0, 0, 0.3)         # Non-interacting electrons with only SOC
    ]

    # (Jphm, g): Orbital exchange + phonon JT coupling strength
    gls = [
        (0.1, 0.1),         # With orbital exchange and JT coupling
        (0, 0),             # No phonon contributions
        (0, 0.1)            # Pure JT coupling (no exchange)
    ]

    Tls = [10, 4]           # Temperatures in Kelvin

    # Return full parameter grid as list of tuples
    # Format: (T, wm, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)
    return [
        (T, 8, N, t, U[0], U[1], g[0], 0.1, g[1], U[2], 24, 5)
        for T in Tls         # Temperature
        for g in gls         # (Jphm, g)
        for U in Usocls      # (U, J, lbd)
        for t in tls         # Hopping
        for N in Nls         # Filling
    ]

