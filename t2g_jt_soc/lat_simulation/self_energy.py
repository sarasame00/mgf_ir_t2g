from maths.ohmatrix import ohmatrix, ohsum
from maths.utils import ohfit

def update_sehf(solver):
    """
    Computes and updates the Hartree-Fock (static) self-energy contribution.
    Based on local Green's function at β (imaginary time).
    """
    gloc_beta = ohsum(solver.irbf.u(solver.beta) * solver.glocl)
    solver._DysonSolver__sehf = ohmatrix(
        -2 * gloc_beta.a * (3 * solver.U - 5 * solver.J),   # diagonal part
         gloc_beta.b * (solver.U - 2 * solver.J)            # off-diagonal (orbital exchange)
    )

def update_sephm(solver):
    """
    Computes and updates the orbital-exchange self-energy (J_phm term).
    This term couples k-space orbital fluctuations with momentum-dependent interaction.
    """
    import numpy as np
    # Generate 3D reciprocal grid
    ky, kx, kz = np.meshgrid(
        *(np.arange(0, 2 * np.pi, 2 * np.pi / solver.k_sz),) * 3
    )

    # Flatten all q-points in Brillouin zone
    qidxs = np.transpose(
        np.indices((solver.k_sz,) * 3), (1, 2, 3, 0)
    ).reshape((solver.k_sz**3, 3))

    # Reset self-energy accumulator
    solver._DysonSolver__sephm *= 0

    # Frequency-summed momentum-resolved Green's function
    gkbeta = ohsum(solver.gkl * solver.irbf.u(solver.beta)[:, None, None, None], axis=0)

    # Loop over all q-points and build interaction kernel
    for qidx in qidxs:
        gqbeta = gkbeta[tuple(qidx)]
        qx, qy, qz = qidx / solver.k_sz * 2 * np.pi
        gammakq = 2 * solver.Jphm * (
            np.cos(kx - qx) + np.cos(ky - qy) + np.cos(kz - qz)
        )
        solver._DysonSolver__sephm += (
            ohmatrix(4 * gqbeta.a, -2 * gqbeta.b) / 3 * gammakq / solver.k_sz**3
        )

def update_se2b(solver):
    """
    Computes and updates second-order local electron-electron self-energy Σ₂^el.
    Follows expressions derived in Supplementary Info Eq. S2.
    """
    a = solver.gloctau.a
    b = solver.gloctau.b
    ab_rev = a[::-1]
    bb_rev = b[::-1]

    # Diagonal (a) component of Σ₂^el in τ
    se2btau_a = (
        (5*solver.U**2 - 20*solver.U*solver.J + 28*solver.J**2)*a**2*ab_rev
        +8*(solver.U**2 - 4*solver.U*solver.J + 3*solver.J**2)*a*b*bb_rev
        -2*(solver.U**2 - 4*solver.U*solver.J + 5*solver.J**2)*b**2*(ab_rev + bb_rev)
    )

    # Off-diagonal (b) component of Σ₂^el in τ
    se2btau_b = (
        (solver.U**2 - 4*solver.U*solver.J + 5*solver.J**2)*a**2*bb_rev
        -2*(solver.U**2 - 2*solver.U*solver.J + 3*solver.J**2)*a*b*(2*ab_rev - bb_rev)
        +(solver.U**2 - 4*solver.U*solver.J + 3*solver.J**2)*b**2*ab_rev
        -(9*solver.U**2 - 36*solver.U*solver.J + 38*solver.J**2)*b**2*bb_rev
    )

    # Fit τ-grid self-energy to IR basis
    solver._DysonSolver__se2bl = ohfit(solver.stauf, ohmatrix(se2btau_a, se2btau_b))

def update_seep(solver):
    """
    Computes and updates electron–phonon self-energy Σ_ep from JT coupling g.
    Based on product of phonon propagator D(τ) and local Green’s function.
    """
    seepftau = -solver.g**2 / 3 * solver.dtau * ohmatrix(
        4/3 * solver.gloctau.a, -2/3 * solver.gloctau.b
    )
    solver._DysonSolver__seepl = ohfit(solver.stauf, seepftau)

def update_seb(solver):
    """
    Computes and updates bosonic (phonon self-energy) correction Σ_B.
    Arises from electron loop contributions to phonon dressing.
    """
    ptau = -4 * (
        solver.gloctau.a * solver.gloctau.a[::-1] -
        solver.gloctau.b * solver.gloctau.b[::-1]
    ) * solver.g**2 / 3
    solver._DysonSolver__sebl = solver.staub.fit(ptau)
