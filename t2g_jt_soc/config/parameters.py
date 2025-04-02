def generate_grid():
    """
    Returns a list of tuples, each containing a parameter set:
    (T, wm, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)
    """
    Nls = [1, 2, 3, 4, 5]
    tls = [0.05, 0.2, 1.2]
    Usocls = [
        (4, 0.8, 0.05),
        (2.5, 0.2, 0.3),
        (0.5, 0.04, 0.3),
        (4, 0.8, 0),
        (2.5, 0.2, 0),
        (0, 0, 0.3)
    ]
    gls = [(0.1, 0.1), (0, 0), (0, 0.1)]
    Tls = [10, 4]

    # Format: (T, wm, N, t, U, J, Jphm, w0, g, lbd, k_sz, diis_mem)
    return [
        (T, 8, N, t, U[0], U[1], g[0], 0.1, g[1], U[2], 24, 5)
        for T in Tls for g in gls for U in Usocls for t in tls for N in Nls
    ]
