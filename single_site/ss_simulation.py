import numpy as np
from scipy.sparse import lil_matrix
from itertools import combinations
import time
import os

def basisFock(num_modes):
    '''
    Function to generate the Fock space basis.
    
    This function constructs all possible occupation states for `num_modes` fermionic orbitals,
    ensuring that each mode follows the Pauli exclusion principle (0 or 1 per mode).
    
    Inputs:
    - num_modes (int): The number of available fermionic orbitals.
    
    Outputs:
    - basis (list of tuples): Each tuple represents an occupation state where elements indicate occupied modes.
    '''
    
    basis = []  
    
    for N in range(num_modes + 1):  # Loop over all possible numbers of occupied modes (from 0 to num_modes)
        basis += list(combinations(range(num_modes), N))  # Generate all possible occupation combinations of N modes
    
    return basis


def annihilationFermi(mode, basis):
    '''
    Function to construct the fermionic annihilation operator.

    This function generates a sparse matrix representation of the fermionic annihilation 
    operator. The operator removes a fermion from the specified mode 
    in the given Fock space basis, following the anti-commutation relations.

    Inputs:
    - mode (int): The index of the mode (orbital) where the annihilation occurs.
    - basis (list of tuples): The Fock space basis, where each tuple represents an occupation state.

    Outputs:
    - A (scipy.sparse.lil_matrix): Sparse matrix representation of the annihilation operator.
    '''
    
    size = len(basis)  # Total number of basis states in the Fock space.
    A = lil_matrix((size, size))  # Initialize a sparse matrix of appropriate size.

    for i, state in enumerate(basis):  # Loop through each basis state.
        if mode in state:  # Annihilation is only possible if the mode is occupied.
            new_state = tuple(x for x in state if x != mode)  # Remove the fermion from the mode.
            sign = (-1) ** state.index(mode)  # Compute the sign due to fermionic anti-commutation rules.
            j = basis.index(new_state)  # Find the index of the new state in the basis list.
            A[j, i] = sign  # Assign the matrix element representing the transition.
    
    return A  # Return the sparse matrix representation of the annihilation operator.


def creationFermi(mode, basis):
    """
    Constructs the fermionic creation operator c†_mode in the given Fock space basis.

    Parameters:
    - mode (int): The index of the fermionic mode where the creation operator acts.
    - basis (list of tuples): The Fock space basis, where each state is a tuple of occupied modes.

    Returns:
    - C (scipy.sparse.lil_matrix): Sparse matrix representation of the creation operator c†_mode.
    """

    size = len(basis)  # Number of basis states
    C = lil_matrix((size, size))  # Initialize a sparse matrix for the operator

    for i, state in enumerate(basis):
        if mode not in state:  # Creation is only possible if mode is unoccupied
            # Create a new state by adding the mode and sorting to maintain order
            new_state = tuple(sorted(state + (mode,)))

            # Compute the sign factor due to fermionic anti-commutation relations
            # (-1)^(number of fermions before mode)
            sign = (-1) ** sum(m < mode for m in state)

            # Find the index of the new state in the basis
            j = basis.index(new_state)

            # Update the creation operator matrix
            C[j, i] = sign

    return C

# Construct the annihilation and creation operators for a 6-mode fermionic system
b = basisFock(6)
ayzu, ayzd, azxu, azxd, axyu, axyd = [annihilationFermi(k, b) for k in range(6)]
cyzu, cyzd, czxu, czxd, cxyu, cxyd = [creationFermi(k, b) for k in range(6)]

nyzu = cyzu * ayzu  # Number operator for orbital component yzu (upper)
nyzd = cyzd * ayzd  # Number operator for orbital component yzd (down)
nyz = nyzu + nyzd   # Total number operator for the nyz orbital

nzxu = czxu * azxu  # Number operator for orbital component zxu (upper)
nzxd = czxd * azxd  # Number operator for orbital component zxd (down)
nzx = nzxu + nzxd   # Total number operator for the nzx orbital

nxyu = cxyu * axyu  # Number operator for orbital component xyu (upper)
nxyd = cxyd * axyd  # Number operator for orbital component xyd (down)
nxy = nxyu + nxyd   # Total number operator for the nxy orbital

Nop = nyz + nzx + nxy  # Total number operator for the system

# Angular Momentum Operators
lx = 1j * (czxu * axyu + czxd * axyd - cxyu * azxu - cxyd * azxd)  # L_x component
ly = 1j * (cxyu * ayzu + cxyd * ayzd - cyzu * axyu - cyzd * axyd)  # L_y component
lz = 1j * (cyzu * azxu + cyzd * azxd - czxu * ayzu - czxd * ayzd)  # L_z component

# Spin Operators
sx = 0.5*(cyzu*ayzd + cyzd*ayzu + czxu*azxd + czxd*azxu + cxyu*axyd + cxyd*axyu)
sy = -1j*0.5*(cyzu*ayzd - cyzd*ayzu + czxu*azxd - czxd*azxu + cxyu*axyd - cxyd*axyu)
sz = 0.5*(cyzu*ayzu - cyzd*ayzd + czxu*azxu - czxd*azxd + cxyu*axyu - cxyd*axyd)


def HK(U, J) :
    '''
    Constructs the Hubbard–Kanamori interaction Hamiltonian for the t₂g shell.

    This function includes:
    - Intra-orbital Coulomb repulsion
    - Inter-orbital Coulomb repulsion (same and opposite spin)
    - Hund’s rule exchange (spin alignment)
    - Pair-hopping processes

    Inputs:
    - U (float): Intra-orbital Coulomb repulsion
    - J (float): Hund's exchange coupling

    Returns:
    - HK (scipy.sparse.lil_matrix): Total interaction Hamiltonian in second quantization
    ''' 
    HK  =  (
        # Intra-orbital Coulomb repulsion: penalizes double occupancy in the same orbital
        U * (nyzu * nyzd + nzxu * nzxd + nxyu * nxyd) +

        # Inter-orbital Coulomb repulsion: penalizes occupation in different orbitals
        (U - 2 * J) * (
            nyzu * nzxd + nyzu * nxyd +  # yz interacting with zx and xy
            nzxu * nyzd + nzxu * nxyd +  # zx interacting with yz and xy
            nxyu * nyzd + nxyu * nzxd    # xy interacting with yz and zx
        ) +

        # Inter-orbital same-spin repulsion: accounts for repulsion between different orbitals
        (U - 3 * J) * (
            nyzu * nzxu + nyzd * nzxd +  # yz and zx interaction
            nyzu * nxyu + nyzd * nxyd +  # yz and xy interaction
            nzxu * nxyu + nzxd * nxyd    # zx and xy interaction
        ) -

        # Hund's exchange interaction: favors parallel spin alignment in different orbitals
        J * (
            cyzu * ayzd * czxd * azxu + cyzu * ayzd * cxyd * axyu +  # yz exchanging with zx and xy
            czxu * azxd * cyzd * ayzu + czxu * azxd * cxyd * axyu +  # zx exchanging with yz and xy
            cxyu * axyd * cyzd * ayzu + cxyu * axyd * czxd * azxu    # xy exchanging with yz and zx
        ) +

        # Pair-hopping term: allows electron pairs to hop between orbitals
        J * (
            cyzu * azxu * cyzd * azxd + cyzu * axyu * cyzd * axyd +  # yz interacting with zx and xy
            czxu * ayzu * czxd * ayzd + czxu * axyu * czxd * axyd +  # zx interacting with yz and xy
            cxyu * ayzu * cxyd * ayzd + cxyu * azxu * cxyd * azxd    # xy interacting with yz and zx
        )
    )
    return HK


def Hsoc(lb):
    '''
    Constructs the spin-orbit coupling (SOC) Hamiltonian for the t₂g shell.

    The SOC term couples spin and orbital angular momenta, lifting the degeneracy of t₂g levels.
    This implementation uses complex combinations of fermionic creation and annihilation operators.

    Parameters:
    - lb (float): Spin-orbit coupling strength λ (in eV)

    Returns:
    - Hsoc (scipy.sparse.lil_matrix): SOC Hamiltonian as a sparse matrix in second quantization
    '''

    Hsoc = 0.5 * lb * (  # Overall SOC prefactor: λ/2
        cyzu * (1j * azxu - axyd) -      # SOC term: yz↑ couples to zx↓ (imaginary) and xy↓ (real)
        cyzd * (1j * azxd - axyu) -      # yz↓ couples to zx↑ and xy↑
        czxu * (1j * ayzu - 1j * axyd) + # zx↑ couples to yz↓ and xy↓
        czxd * (1j * ayzd + 1j * axyu) + # zx↓ couples to yz↑ and xy↑
        cxyu * (ayzd - 1j * azxd) -      # xy↑ couples to yz↓ and zx↓
        cxyd * (ayzu + 1j * azxu)        # xy↓ couples to yz↑ and zx↑
    )

    return Hsoc  # Return the spin-orbit Hamiltonian as a sparse matrix


def Hjt(g, qx, qy):
    '''
    Constructs the Jahn-Teller (JT) interaction Hamiltonian for t₂g orbitals.

    The JT effect couples lattice distortions (Qx, Qy) to orbital occupations,
    lowering symmetry and lifting orbital degeneracy in a way that depends on
    orbital imbalance.

    Parameters:
    - g (float): Jahn-Teller coupling constant
    - qx (float): Jahn-Teller distortion component along Q₂ mode (orthorhombic)
    - qy (float): Jahn-Teller distortion component along Q₃ mode (tetragonal)

    Returns:
    - Hjt (scipy.sparse.lil_matrix): Jahn-Teller interaction Hamiltonian in second quantization
    '''

    return g * (  
        qy * (nyz - nzx) / np.sqrt(3) +      # Orthorhombic Q₃ mode: imbalance between yz and zx
        qx * (2 * Nop / 3 - nyz - nzx)       # Tetragonal Q₂ mode: deviation from spherical symmetry
    )


def eigobj(x, U, J, lb, N, B, g):
    """
    Compute the lowest eigenvalue of the Hamiltonian for a given Jahn-Teller distortion (Qx, Qy),
    restricted to the subspace with total particle number N.

    The Hamiltonian includes:
    - Kanamori interaction
    - Spin-orbit coupling
    - Jahn-Teller distortion
    - Harmonic confinement potential

    Parameters:
    - x (tuple): (Qx, Qy) distortion coordinates
    - U (float): Intra-orbital Coulomb repulsion
    - J (float): Hund's exchange coupling
    - lb (float): Spin-orbit coupling strength
    - N (int): Total electron count
    - B (float): Harmonic confinement strength
    - g (float): Jahn-Teller coupling

    Returns:
    - float: Ground state energy (lowest eigenvalue + harmonic potential)
    """

    # Mapping from particle number N to Hamiltonian subspace slice
    sl = {
        0: slice(0, 1),
        1: slice(1, 7),
        2: slice(7, 22),
        3: slice(22, 42),
        4: slice(42, 57),
        5: slice(57, 63),
        6: slice(63, 64)
    }

    # Build full Hamiltonian and extract N-particle subspace
    H_full = HK(U, J) + Hsoc(lb) + Hjt(g, x[0], x[1])
    HN = H_full[sl[N], sl[N]]

    # Compute eigenvalues in the N-particle subspace
    w = np.linalg.eigvalsh(HN.toarray())

    # Add harmonic confinement energy to the lowest eigenvalue
    return w[0] + 0.5 * B * (x[0]**2 + x[1]**2)


