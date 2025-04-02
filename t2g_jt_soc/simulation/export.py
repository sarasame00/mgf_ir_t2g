import h5py
import numpy as np

from maths.ohmatrix import ohsum, ohmatrix
from maths.utils import ohfit

def export_solver_to_hdf5(solver, path):
    """
    Exports all relevant quantities from a solved DysonSolver instance to an HDF5 file.
    This includes input parameters, Green's functions, self-energies, observables, and correlators.

    Parameters:
    - solver: A fully solved DysonSolver object containing all model state and computed quantities.
    - path (str): Destination path (without extension) for the .hdf5 file.
    """
    with h5py.File(path + '.hdf5', 'w') as f:

        # ====================
        # ⬇ Model parameters and metadata
        # ====================

        f.create_dataset("T", data = solver.T)                          # Temperature (Kelvin)
        f.create_dataset("beta", data = solver.beta)                    # Inverse temperature β = 1/kT
        f.create_dataset("wmax", data = solver.wM)                      # IR basis cutoff frequency (Λ)
        f.create_dataset("N", data = solver.N)                          # Target particle number
        f.create_dataset("nexp", data = -6 * np.sum(
            solver.irbf.u(solver.beta) * solver.glocl.a))               # Computed density estimate

        f.create_dataset("t", data = solver.t)                          # Hopping amplitude
        f.create_dataset("U", data = solver.U)                          # On-site Coulomb interaction
        f.create_dataset("J", data = solver.J)                          # Hund's coupling
        f.create_dataset("Jphm", data = solver.Jphm)                    # Orbital exchange interaction
        f.create_dataset("Up", data = solver.Up)                        # U' = U - 2J
        f.create_dataset("w0", data = solver.w0)                        # Phonon frequency
        f.create_dataset("g", data = solver.g)                          # Jahn-Teller electron-phonon coupling
        f.create_dataset("lbd", data = solver.lbd)                      # Spin-orbit coupling (λ)
        f.create_dataset("k_sz", data = solver.k_sz)                    # Number of k-points in each direction
        f.create_dataset("diis_mem", data = solver.diis_mem)            # DIIS extrapolation memory size
        f.create_dataset("mu", data = solver.mu)                        # Chemical potential at convergence
        f.create_dataset("conv", data = np.array(solver.conv_ls))    # Convergence values over iterations


        # ====================
        # ⬇ Green’s functions and self-energy components
        # ====================

        f.create_dataset("glocl_a", data = solver.glocl.a)              # Local Green's function (diagonal)
        f.create_dataset("glocl_b", data = solver.glocl.b)              # Local Green's function (off-diagonal)

        # (commented out: full k-resolved Green’s functions can be large)
        # f.create_dataset("gkl_a", data = solver.gkl.a)
        # f.create_dataset("gkl_b", data = solver.gkl.b)

        f.create_dataset("dl", data = solver.dl)                        # Phonon propagator D(iωₙ)
        f.create_dataset("sehf_a", data = solver.sehf.a)                # Hartree-Fock self-energy (a part)
        f.create_dataset("sehf_b", data = solver.sehf.b)                # Hartree-Fock self-energy (b part)
        f.create_dataset("seepl_a", data = solver.seepl.a)              # e-ph self-energy (a)
        f.create_dataset("seepl_b", data = solver.seepl.b)              # e-ph self-energy (b)
        f.create_dataset("sebl", data = solver.sebl)                    # Bosonic self-energy (phonon dressing)


        # ====================
        # ⬇ Energy contributions (per β)
        # ====================

        # Spin-orbit potential energy: Tr[G * V] ∝ λ ⟨l·s⟩
        f.create_dataset("epot", data = -0.5 * solver.lbd * (
            ohsum(solver.irbf.u(solver.beta) * solver.glocl) * ohmatrix(0, 1)
        ).trace)

        # Electron-phonon interaction energy (Tr[G * Σ_eph])
        Fepl = ohfit(solver.smatf, solver.glociw * solver.seepiw).real
        Fepbeta = ohsum(solver.irbf.u(solver.beta) * Fepl)
        f.create_dataset("eeph", data = -Fepbeta.trace)

        # Electron-electron interaction energy (Tr[G * Σ_HF + Σ_2])
        Feel = ohfit(solver.smatf, solver.glociw * (solver.sehf + solver.se2biw)).real
        Feebeta = ohsum(solver.irbf.u(solver.beta) * Feel)
        f.create_dataset("eint", data = -Feebeta.trace)

        # Orbital exchange energy contribution (Tr[G_k * Σ_phm])
        Fphml = ohfit(
            solver.smatf,
            ohsum(solver.gkiw * solver.sephm[None, :, :, :], axis=(1, 2, 3)) / solver.k_sz**3
        ).real
        f.create_dataset("ephm", data = -ohsum(Fphml * solver.irbf.u(solver.beta)).trace)

        # Kinetic energy: Tr[H_k * G_k]
        gkbeta = ohsum(solver.irbf.u(solver.beta)[:, None, None, None] * solver.gkl, axis=0)
        f.create_dataset("ekin", data = -ohsum(gkbeta * solver.Hlatt).trace / solver.k_sz**3)

        # Electron chemical energy: Tr[μ * G]
        f.create_dataset("eche", data = -ohsum(solver.irbf.u(solver.beta) * solver.glocl).trace * solver.mu)

        # Phonon energy: ⟨n_ph⟩ * ω₀
        f.create_dataset("ephn", data = -2 * np.sum(solver.irbb.u(solver.beta) * solver.dl) * solver.w0)
        f.create_dataset("nph", data = -2 * np.sum(solver.irbb.u(solver.beta) * solver.dl))
        f.create_dataset("nphel", data = solver.nph0)  # Reference phonon population (non-interacting)

        # Total energy: sum of all contributions
        f.create_dataset("etot", data =
            f["epot"][()] + f["eeph"][()] + f["ephm"][()] +
            f["eint"][()] + f["ekin"][()] + f["eche"][()] + f["ephn"][()]
        )


        # ====================
        # ⬇ Local density variance estimate (used to track fluctuations)
        # ====================
        gloc0 = ohsum(solver.irbf.u(0) * solver.glocl)
        glocf = ohsum(solver.irbf.u(solver.beta) * solver.glocl)
        var = np.sqrt(12 * gloc0.a * glocf.a + 6 * gloc0.b * glocf.b)
        f.create_dataset("varn", data=var)


        # ====================
        # ⬇ Orbital correlations in the irreducible Brillouin zone
        # ====================

        # Get momentum points in full and irreducible Brillouin zone
        kidxs = np.transpose(np.indices((solver.k_sz,) * 3), (1, 2, 3, 0)).reshape((-1, 3))
        qidxs = kidxs[np.where(
            (kidxs[:, 0] <= solver.k_sz // 2) &
            (kidxs[:, 1] <= kidxs[:, 0]) &
            (kidxs[:, 2] <= kidxs[:, 0]) &
            (kidxs[:, 2] <= kidxs[:, 1])
        )]

        # Initialize correlation observables
        corrdiag = np.zeros(qidxs.shape[0])
        corroffd = np.zeros(qidxs.shape[0])
        corrcrs1 = np.zeros(qidxs.shape[0])
        corrcrs2 = np.zeros(qidxs.shape[0])

        for i, q in enumerate(qidxs):
            # Apply q-shift to momentum grid and wrap (periodic)
            kidxst = (kidxs - q) % solver.k_sz
            kidxstf = solver.k_sz**2 * kidxst[:, 0] + solver.k_sz * kidxst[:, 1] + kidxst[:, 2]

            # Convolution in k-space: G(k - q)
            gklconv = solver.gkl.reshape((solver.irbf.size, -1))[:, kidxstf].reshape((solver.irbf.size,) + (solver.k_sz,) * 3)
            gkzero = ohsum(solver.irbf.u(0)[:, None, None, None] * solver.gkl, axis=0)
            gkconvbeta = ohsum(solver.irbf.u(solver.beta)[:, None, None, None] * gklconv, axis=0)

            # Accumulate correlation components
            corrdiag[i] = np.sum(gkzero.a * gkconvbeta.a) / solver.k_sz**3         # Diagonal channel
            corroffd[i] = np.sum(gkzero.b * gkconvbeta.b) / solver.k_sz**3         # Off-diagonal channel
            corrcrs1[i] = -np.sum(gkzero.a * gkconvbeta.b) / solver.k_sz**3        # Cross-term 1
            corrcrs2[i] = -np.sum(gkzero.b * gkconvbeta.a) / solver.k_sz**3        # Cross-term 2

        # Save correlation data
        f.create_dataset("correldiag", data=corrdiag)
        f.create_dataset("correloffd", data=corroffd)
        f.create_dataset("correlcrs1", data=corrcrs1)
        f.create_dataset("correlcrs2", data=corrcrs2)
        f.create_dataset("irrBZ", data=qidxs)  # Stored q-points in irreducible zone
