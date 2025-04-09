import numpy as np

def rot_mat(theta, u):
    """
    Computes the integer-valued 3×3 rotation matrix for a rotation of angle `theta` (in degrees)
    around an arbitrary axis `u`, using the Rodrigues rotation formula.

    Parameters:
    - theta (float): Rotation angle in degrees.
    - u (array-like): 3D axis vector to rotate around (will be normalized internally).

    Returns:
    - np.ndarray (3x3 int): Rounded integer rotation matrix.
    """

    # Normalize the rotation axis
    u = u / np.sqrt(np.dot(u, u))

    # Convert angle from degrees to radians
    theta *= np.pi / 180

    # Build skew-symmetric matrix K for cross-product operator: K·v = u × v
    K = np.array([
        np.cross([1, 0, 0], u),  # row corresponding to x
        np.cross([0, 1, 0], u),  # row corresponding to y
        np.cross([0, 0, 1], u)   # row corresponding to z
    ])

    # Rodrigues' rotation formula:
    # R = I + sin(θ)·K + (1 - cos(θ))·K²
    # This gives a rotation matrix around vector `u` by angle `θ`
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # Return rounded version as integer matrix (used for discrete symmetry operations)
    return np.round(R).astype(np.int32)


# ------------------------------------------------------------------------------
# Ohsym: List of rotation matrices forming the full **Oh point group** (cubic symmetry group)
# ------------------------------------------------------------------------------

# Each element is a 3×3 integer matrix representing a symmetry operation
# in the cubic group (Oh), generated using the Rodrigues formula in `rot_mat`.

Ohsym = [
    rot_mat(0,   (1, 0, 0)),   # Identity

    # 180° rotations around principal axes (x, y, z)
    rot_mat(180, (1, 0, 0)),   
    rot_mat(180, (0, 1, 0)),
    rot_mat(180, (0, 0, 1)),

    # 180° rotations around face diagonals
    rot_mat(180, (1, 1, 0)),
    rot_mat(180, (-1, 1, 0)),
    rot_mat(180, (1, 0, 1)),
    rot_mat(180, (1, 0, -1)),
    rot_mat(180, (0, -1, -1)),
    rot_mat(180, (0, -1, 1)),

    # 90° rotations around principal axes (x, y, z)
    rot_mat(90, (1, 0, 0)),
    rot_mat(90, (0, 1, 0)),
    rot_mat(90, (0, 0, 1)),

    # 270° = -90° rotations (same axes, opposite direction)
    rot_mat(90, (-1, 0, 0)),
    rot_mat(90, (0, -1, 0)),
    rot_mat(90, (0, 0, -1)),

    # 120° rotations around body diagonals (e.g., (1,1,1) axis)
    rot_mat(120, (1, 1, 1)),
    rot_mat(120, (-1, 1, 1)),
    rot_mat(120, (1, -1, 1)),
    rot_mat(120, (1, 1, -1)),
    rot_mat(120, (1, -1, -1)),
    rot_mat(120, (-1, 1, -1)),
    rot_mat(120, (-1, -1, 1)),
    rot_mat(120, (-1, -1, -1)),
]

# ------------------------------------------------------------------------------
# Include improper rotations (inversion through the origin)
# by adding the negatives of all Oh symmetry matrices
# This completes the full **O_h** point group (including inversion symmetry)
# ------------------------------------------------------------------------------

Ohsym += [-s for s in Ohsym]  # Adds -R for each rotation R in Ohsym


def antifourier(k_sz, irrBZ, func, xmin=0, xmax=16, axis=0):
    """
    Computes the real-space correlation profile from momentum-space data 
    via an inverse Fourier transform over a symmetry-expanded Brillouin zone.

    Parameters:
    ----------
    k_sz : int
        Size of the k-point grid along each axis (k-space cube is k_sz³).
    irrBZ : np.ndarray
        Array of shape (n_q, 3) with momentum vectors (q) in the irreducible Brillouin zone.
    func : np.ndarray
        Correlation values at each q-point in irrBZ (shape = (n_q,)).
    xmin, xmax : int
        Range of real-space distances to compute over (in lattice units).
    axis : int
        Which axis to perform the projection onto (0=x, 1=y, 2=z).

    Returns:
    -------
    xsites : np.ndarray
        Array of real-space distances (integers from xmin to xmax).
    profile : np.ndarray
        Real-space correlation profile G(x), computed via inverse Fourier transform:
        G(x) = ∑_k e^{-i k·x} f(k)
    """

    # Real-space distance array: x = [xmin, ..., xmax-1]
    xsites = np.arange(xmin, xmax)

    # Generate full grid of k-point indices (shape = (k_sz, k_sz, k_sz, 3))
    k_idxs = np.transpose(np.indices((k_sz,) * 3), (1, 2, 3, 0))

    # Construct inverse Fourier kernel: e^{-i k · x} (projected along chosen axis)
    # shape: (k_sz, k_sz, k_sz, len(xsites))
    euler_factor = np.exp(
        -1j * 2 * np.pi * k_idxs[:, :, :, axis, None] * xsites[None, None, None, :]
        / k_sz
    )

    # Initialize the function over the full Brillouin zone
    func_expand = np.zeros((k_sz, k_sz, k_sz))

    # Symmetry expansion: fill full BZ using Oh symmetry operations
    for i, q in enumerate(irrBZ):
        for s in Ohsym:
            kx, ky, kz = (s @ q) % k_sz  # Rotate and wrap around periodic BZ
            func_expand[kx, ky, kz] = func[i]

    # Perform the sum over k-space to get real-space correlation profile
    # Result shape: (len(xsites),)
    profile = np.sum(euler_factor * func_expand[:, :, :, None], axis=(0, 1, 2)).real
    profile /= k_sz**3  # Normalize by total number of k-points

    return xsites, profile


def take_borders(irrBZ, func):
    """
    Extracts a high-symmetry path (Γ → X → M → R) in the irreducible Brillouin zone
    and interpolates the associated correlator or spectral function values.

    Parameters:
    ----------
    irrBZ : np.ndarray of shape (n_q, 3)
        Momentum points in the irreducible Brillouin zone.
    func : np.ndarray of shape (n_q,)
        Quantity defined at each momentum point (e.g., a correlation function).

    Returns:
    -------
    dist_path : np.ndarray
        Cumulative distance along the high-symmetry path (in units of 2π / a).
    func_path : np.ndarray
        The values of `func` interpolated along the same path.
    """

    # Approximate total size of the full Brillouin zone (assuming irrBZ spans up to k_sz/2)
    k_sz = np.max(irrBZ) * 2

    # Identify path segments by their constraints in k-space:
    GXpath = np.where((irrBZ[:, 1] == 0) & (irrBZ[:, 2] == 0))               # Γ → X
    XMpath = np.where((irrBZ[:, 0] == k_sz // 2) & (irrBZ[:, 2] == 0))       # X → M
    GMpath = np.where((irrBZ[:, 0] == irrBZ[:, 1]) & (irrBZ[:, 2] == 0))     # Γ → M
    GRpath = np.where((irrBZ[:, 0] == irrBZ[:, 1]) & (irrBZ[:, 2] == irrBZ[:, 1]))  # Γ → R
    XRpath = np.where((irrBZ[:, 0] == k_sz // 2) & (irrBZ[:, 2] == irrBZ[:, 1]))    # X → R
    MRpath = np.where((irrBZ[:, 1] == k_sz // 2) & (irrBZ[:, 0] == k_sz // 2))      # M → R

    # Concatenate the values of `func` along the full path:
    func_path = np.concatenate((
        func[GXpath][:-1],
        func[XMpath][:-1],
        func[GMpath][:0:-1],
        func[GRpath][:-1],
        func[XRpath][:0:-1],
        func[XMpath][:-1],
        func[MRpath]
    ))

    # Concatenate the associated momentum points in same order:
    irrBZ_path = np.concatenate((
        irrBZ[GXpath][:-1],
        irrBZ[XMpath][:-1],
        irrBZ[GMpath][:0:-1],
        irrBZ[GRpath][:-1],
        irrBZ[XRpath][:0:-1],
        irrBZ[XMpath][:-1],
        irrBZ[MRpath]
    ))

    # Compute cumulative distance along the path
    delta_path = np.concatenate((
        np.array([0]),
        np.sqrt(np.sum((irrBZ_path[1:] - irrBZ_path[:-1])**2, axis=1))
    ))

    # Cumulative sum of distances (via matrix product for efficiency)
    A = np.tril(np.ones(delta_path.size))
    dist_path = A @ delta_path

    # Normalize by grid size and convert to physical units (2π / a)
    return dist_path / k_sz * 2 * np.pi, func_path
