import numpy as np
from scipy.sparse import lil_matrix
from itertools import combinations

# === Parameters ===

U = 2.5  # Hubbard intra-orbital Coulomb interaction
J = 0.2  # Hund's exchange coupling
g = 0.1        # Coupling constant
B = 0.1        # Magnetic field strength
size_grid = 101 # Grid size for numerical calculations
Qmax = 1.2      # Maximum value for a parameter Q
N_values = range(1, 6)
lb_values = [0, 0.1, 0.3, 0.5] # Values of spin-orbit coupling strength (λ)


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
    - C (lil_matrix): Sparse matrix representation of the creation operator c†_mode.
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


# Define the Hubbard-Kanamori interaction Hamiltonian (HK)
def HK(U, J) : 
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

# Define the spin-orbit coupling Hamiltonian (Hsoc)
def Hsoc(lb):
    """
    Computes the spin-orbit coupling (SOC) Hamiltonian.
    
    Parameters:
    - lb (float): The spin-orbit coupling strength (λ).
    
    Returns:
    - Hsoc (matrix): The SOC Hamiltonian matrix.
    """
    Hsoc = 0.5 * lb * (
        # Terms coupling spin and orbital motion with complex phase factors
        cyzu * (1j * azxu - axyd) -  # Coupling in yz orbital (up)
        cyzd * (1j * azxd - axyu) -  # Coupling in yz orbital (down)
        czxu * (1j * ayzu - 1j * axyd) +  # Coupling in zx orbital (up)
        czxd * (1j * ayzd + 1j * axyu) +  # Coupling in zx orbital (down)
        cxyu * (ayzd - 1j * azxd) -  # Coupling in xy orbital (up)
        cxyd * (ayzu + 1j * azxu)  # Coupling in xy orbital (down)
    )
    return Hsoc

# Define the Jahn-Teller interaction Hamiltonian in terms of Qx and Qy
def Hjt(qx, qy): 
    return g * (qy * (nyz - nzx) / np.sqrt(3) +  # Contribution along yz and zx orbitals
        qx * (2 * Nop / 3 - nyz - nzx)   # Contribution affecting total electron number
)

# Function to compute the lowest eigenvalue of the Hamiltonian in a selected subspace
def eigobj(x, N=1):
    """
    Computes the lowest eigenvalue of the Hamiltonian HN for a given mode x and subspace N.

    Parameters:
    x : tuple
        (x[0], x[1]) represent the parameters for the Jahn-Teller interaction.
    N : int
        Index selecting a particular subspace in the Hamiltonian.

    Returns:
    float : The lowest eigenvalue of HN with an additional harmonic trapping term.
    """

    # Define index slices for different orbital/mode components
    sl = {
        0: slice(0, 1),   # First range: single element at index 0
        1: slice(1, 7),   # Second range: elements from index 1 to 6
        2: slice(7, 22),  # Third range: elements from index 7 to 21
        3: slice(22, 42), # Fourth range: elements from index 22 to 41
        4: slice(42, 57), # Fifth range: elements from index 42 to 56
        5: slice(57, 63), # Sixth range: elements from index 57 to 62
        6: slice(63, 64)  # Last range: single element at index 63
    }

    # Extract the submatrix corresponding to the selected subspace
    HN = (HK(U, J) + Hsoc(lb) + Hjt(x[0], x[1]))[sl[N], sl[N]]

    # Compute eigenvalues of HN
    w = np.linalg.eigvalsh(HN.toarray())

    # Return the lowest eigenvalue with an additional quadratic trapping term
    return w[0] + 0.5 * B * (x[0]**2 + x[1]**2)


# Loop over different values of spin-orbit coupling strength (λ)
for lb in lb_values:
    print('-' * 5)  # Print a separator for readability

    # Define a 2D grid for Qx and Qy ranging from -Qmax to Qmax
    Qx, Qy = np.meshgrid(*(np.linspace(-Qmax, Qmax, size_grid),) * 2)

    # Loop over different subspaces N
    for N in N_values:
        print(f'λ = {lb}, N = {N}')  # Print the current subspace index

        # Initialize an energy map for the given subspace
        emap = np.zeros((size_grid, size_grid))

        # Loop over the grid points
        for i in range(size_grid):
            for j in range(size_grid):
                # Compute the lowest eigenvalue of the Hamiltonian for each (Qx, Qy) point
                emap[i,j] = eigobj([Qx[i, j], Qy[i, j]], N)
        # Normalize the energy map by subtracting its minimum value
        emap -= np.min(emap)

        # Save the computed data as a text file for later use in a paper figure
        np.savetxt(
            "PaperFigs/T0data/%iN_%iSOC_lowee.txt" % (N, 10 * lb),
            np.array((Qx, Qy, emap)).reshape((3, emap.shape[0] * emap.shape[1])).T
        )

# Print separator to indicate the end of the computation
print('-' * 15 + '\n')