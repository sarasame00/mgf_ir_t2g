import numpy as np
import sparse_ir as ir

from maths.ohmatrix import ohmatrix, ohsum, ohcopy, ohzeros
from maths.utils import fprint, ohfit, ohevaluate
from lat_simulation import self_energy
from lat_simulation.export import export_solver_to_hdf5


class DysonSolver:
    """
    Self-consistent Dyson equation solver for a multi-orbital, spin-orbit coupled system 
    with local Coulomb and Jahn-Teller interactions, using sparse IR basis.

    This class performs:
    - Construction of G₀ and full G via Dyson equation
    - Update of Hartree, second-order, phonon, and SOC self-energies
    - DIIS acceleration and convergence tracking
    - Export of results to HDF5
    """

    def __init__(self, T, wM, N, t, U, J, Jphm, w0, g, lbd, sz, diis_mem, fl="t2g_soc_jtpol.out"):
        """
        Initialize all model parameters, grids, basis, and internal state.

        Parameters:
        - T: temperature (Kelvin)
        - wM: Matsubara frequency cutoff
        - N: target particle number
        - t: hopping amplitude
        - U: Coulomb repulsion
        - J: Hund's coupling
        - Jphm: orbital exchange interaction
        - w0: phonon frequency
        - g: JT electron-phonon coupling
        - lbd: spin-orbit coupling strength (λ)
        - sz: number of k-points per axis
        - diis_mem: DIIS extrapolation memory
        - fl: output file path for logging
        """
        # Store input parameters
        self.__T = T
        self.__wM = wM
        self.__N = N
        self.__t = t
        self.__U = U
        self.__J = J
        self.__Jphm = Jphm
        self.__w0 = w0
        self.__g = g
        self.__lbd = lbd
        self.__diis_mem = diis_mem
        self.__fl = fl

        # Construct the 3D tight-binding dispersion on a cubic lattice
        ky, kx, kz = np.meshgrid(*(np.arange(0, 2*np.pi, 2*np.pi/sz),)*3)
        self.__Hlatt = -2 * t * (np.cos(kx) + np.cos(ky) + np.cos(kz))

        # Setup sparse IR basis for fermionic (electrons) and bosonic (phonons) sectors
        self.__irbf = ir.FiniteTempBasis('F', self.beta, wM)
        self.__stauf = ir.TauSampling(self.__irbf)
        self.__smatf = ir.MatsubaraSampling(self.__irbf)
        self.__freqf = 1j * self.__smatf.wn * np.pi / self.beta

        self.__irbb = ir.FiniteTempBasis('B', self.beta, wM)
        self.__staub = ir.TauSampling(self.__irbb)
        self.__smatb = ir.MatsubaraSampling(self.__irbb)
        self.__freqb = 1j * self.__smatb.wn * np.pi / self.beta

        # Initialize fields (Green's functions and self-energies)
        self.__mu = 0
        self.__sehf = ohmatrix(0, 0)
        self.__sephm = ohzeros((sz, sz, sz))
        self.__seepl = ohzeros(self.__irbf.size)
        self.__se2bl = ohzeros(self.__irbf.size)
        self.__sebl = np.zeros(self.__irbb.size)
        self.__glocl = ohzeros(self.__irbf.size)
        self.__gkl = ohzeros((self.__irbf.size, sz, sz, sz))
        self.__dl = 0  # phonon propagator in IR

        # Prepare DIIS memory and convergence trackers
        self.__conv_ls = []
        self.__diis_vals = ohzeros(1) if diis_mem == 0 else ohzeros((diis_mem, self.irbf.size+1))
        self.__diis_err = ohzeros(1) if diis_mem == 0 else ohzeros((diis_mem, self.irbf.size+1))

        self.__solved = False

        # Precompute initial bosonic propagator and non-interacting phonon occupation
        self.__update_gb()
        self.__nph0 = -2 * np.sum(self.irbb.u(self.beta) * self.dl)

    # ------------- Properties: convenience access to private attributes -------------

    @property
    def T(self): return self.__T
    @property
    def beta(self): return 11604.522110519543 / self.__T
    @property
    def wM(self): return self.__wM
    @property
    def N(self): return self.__N
    @property
    def t(self): return self.__t
    @property
    def U(self): return self.__U
    @property
    def Up(self): return self.__U - 2 * self.__J
    @property
    def J(self): return self.__J
    @property
    def Jphm(self): return self.__Jphm
    @property
    def w0(self): return self.__w0
    @property
    def g(self): return self.__g
    @property
    def lbd(self): return self.__lbd
    @property
    def k_sz(self): return self.__Hlatt.shape[0]
    @property
    def diis_mem(self): return self.__diis_mem

    # Model structure and sampling basis
    @property
    def Hlatt(self): return self.__Hlatt
    @property
    def irbf(self): return self.__irbf
    @property
    def smatf(self): return self.__smatf
    @property
    def stauf(self): return self.__stauf
    @property
    def freqf(self): return self.__freqf
    @property
    def irbb(self): return self.__irbb
    @property
    def smatb(self): return self.__smatb
    @property
    def staub(self): return self.__staub
    @property
    def freqb(self): return self.__freqb

    # Field variables and self-energies
    @property
    def mu(self): return self.__mu
    @property
    def sehf(self): return self.__sehf
    @property
    def sephm(self): return self.__sephm
    @property
    def se2bl(self): return self.__se2bl
    @property
    def se2btau(self): return ohevaluate(self.__stauf, self.se2bl)
    @property
    def se2biw(self): return ohevaluate(self.__smatf, self.se2bl)
    @property
    def seepl(self): return self.__seepl
    @property
    def seeptau(self): return ohevaluate(self.__stauf, self.seepl)
    @property
    def seepiw(self): return ohevaluate(self.__smatf, self.seepl)
    @property
    def sebl(self): return self.__sebl
    @property
    def sebtau(self): return self.__staub.evaluate(self.sebl)
    @property
    def sebiw(self): return self.__smatb.evaluate(self.sebl)

    # Green's functions
    @property
    def glocl(self): return self.__glocl
    @property
    def gloctau(self): return ohevaluate(self.__stauf, self.glocl)
    @property
    def glociw(self): return ohevaluate(self.__smatf, self.glocl)
    @property
    def gkl(self): return self.__gkl
    @property
    def gktau(self): return ohevaluate(self.__stauf, self.gkl, axis=0)
    @property
    def gkiw(self): return ohevaluate(self.__smatf, self.gkl, axis=0)

    # Bosonic propagator
    @property
    def dl(self): return self.__dl
    @property
    def dtau(self): return self.__staub.evaluate(self.dl)
    @property
    def diw(self): return self.__smatb.evaluate(self.dl)

    @property
    def conv_ls(self): return self.__conv_ls
    @property
    def nph0(self): return self.__nph0

    # ------------- Green's function / propagator updates -------------

    def __update_green(self, out_fl, tol=1e-6, delta=0.1):
        """
        Self-consistently adjusts the chemical potential μ to enforce the target electron density N.

        This is done via fixed-point iteration: μ is updated based on the deviation between
        the computed electron density and the target value. The solver uses IR basis projection
        and Matsubara Green's functions to estimate the density.

        Parameters:
        - out_fl (file): Open file object for writing logs (usually .out file).
        - tol (float): Desired tolerance for density convergence (|N - N_exp|).
        - delta (float): Initial step size for adjusting μ.
        """
        
        # Estimate an initial guess for μ using the trace of the self-energy
        self.__mu = np.sum((
            self.sehf + ohsum(self.irbf.u(self.beta) * (self.se2bl + 2 * self.seepl))
        ).real.eigvals) / 2

        last_sign = 0  # To track whether we're increasing or decreasing μ

        while True:
            fprint("Starting with mu=%.8f" % self.mu, out_fl)

            if self.__t != 0:
                # Full k-resolved Green's function G_k(iωₙ)
                gkiw = (
                    self.freqf[:, None, None, None] - self.Hlatt[None, :, :, :]  # ε_k
                    - self.sehf                                                   # Static HF self-energy
                    - self.sephm[None, :, :, :]                                   # Momentum-dependent self-energy
                    - 2 * self.seepiw[:, None, None, None]                        # Dynamical e-ph self-energy
                    + self.mu                                                     # Chemical potential
                    - 0.5 * self.lbd * ohmatrix(0, 1)                              # Spin-orbit term
                )**-1

                # Project G_k(iωₙ) to IR basis using ohfit and store only real part
                self.__gkl = ohfit(self.smatf, gkiw, axis=0).real

                # Compute local Green’s function G_loc(iωₙ) by momentum averaging
                glociw = ohsum(gkiw, axis=(1, 2, 3)) / self.k_sz**3

            else:
                # In atomic limit (no hopping): only local Green's function
                glociw = (
                    self.freqf
                    - self.sehf
                    - 2 * self.seepiw
                    + self.mu
                    - 0.5 * self.lbd * ohmatrix(0, 1)
                )**-1

            # Project G_loc(iωₙ) into the IR basis
            self.__glocl = ohfit(self.smatf, glociw).real

            # Estimate density using trace over G_loc(τ=β)
            Nexp = -6 * np.sum(self.irbf.u(self.beta) * self.glocl.a)
            fprint("Finished with Nexp=%.8f" % (Nexp.real), out_fl)

            # Difference from target density
            DN = self.N - Nexp

            # If density matches within tolerance → stop iteration
            if abs(DN) <= tol:
                return

            # Otherwise, adapt μ (and shrink delta if sign flipped)
            if DN > 0:
                if last_sign == -1:
                    delta /= 2
                self.__mu += delta
                last_sign = +1
            elif DN < 0:
                if last_sign == +1:
                    delta /= 2
                self.__mu -= delta
                last_sign = -1


    def __update_gb(self):
        """
        Updates the interacting bosonic (phonon) propagator D(iωₙ) in Matsubara frequency space.

        This is done by dressing the bare propagator D₀(iωₙ) with the bosonic self-energy Σ_b(iωₙ),
        using Dyson's equation for bosons:

            D(iωₙ) = [D₀⁻¹(iωₙ) - Σ_b(iωₙ)]⁻¹

        The result is then fitted into the IR basis and stored in self.__dl.
        """
        
        # Compute the bare propagator D₀(iωₙ) for Einstein phonons
        # D₀(iωₙ) = 2ω₀ / (ωₙ² - ω₀²)
        d0iw = (2 * self.w0 / (self.freqb**2 - self.w0**2)).real

        # Compute dressed propagator via Dyson equation: D = [D₀⁻¹ - Σ_b]⁻¹
        diw = (d0iw**(-1) - self.sebiw)**(-1)

        # Fit the result into the bosonic IR basis
        self.__dl = self.smatb.fit(diw).real


    # ------------- Main solver loop -------------

    def solve(self, diis_active=True, tol=1e-6, max_iter=10000):
        """
        Perform the full self-consistent Dyson equation solver loop.

        This iterative process updates the Green's function and self-energies until convergence.
        Optionally uses Direct Inversion of the Iterative Subspace (DIIS) to accelerate convergence.

        Parameters:
        - diis_active (bool): Whether to use DIIS extrapolation to accelerate convergence.
        - tol (float): Convergence threshold (based on change in gloc).
        - max_iter (int): Maximum number of SCF iterations allowed.
        """
        
        # Disable DIIS explicitly if memory is set to zero
        if self.diis_mem == 0:
            diis_active = False

        # Open log file for writing output (e.g., convergence messages)
        out_fl = open(self.__fl, 'w')

        # Log simulation parameters
        fprint("Starting execution with the following parameters", file=out_fl)
        for name in ["T", "beta", "wM", "N", "t", "U", "J", "Jphm", "w0", "g", "lbd", "k_sz", "diis_mem"]:
            fprint(f"{name}={getattr(self, name):.3f}", file=out_fl)
        fprint("-" * 15 + "\n", file=out_fl)

        # Initial guess for non-interacting Green's function and phonon propagator
        fprint("Computing non-interacting Green's function", file=out_fl)
        self.__update_green(out_fl)
        self.__update_gb()
        fprint('\n'*2, file=out_fl)

        iterations = 0  # SCF iteration counter

        while True:
            # Backup current Green's function to compare for convergence
            last_g = ohcopy(self.glocl)

            fprint(f"Starting iteration {iterations + 1}", file=out_fl)
            fprint("Updating self-energies", file=out_fl)

            # Compute all components of the total self-energy Σ = Σ_HF + Σ_phm + Σ_2B + Σ_eph + Σ_B
            self_energy.update_sehf(self)
            self_energy.update_sephm(self)
            self_energy.update_se2b(self)
            self_energy.update_seep(self)
            self_energy.update_seb(self)

            # =============== DIIS acceleration block ===============
            if diis_active:
                # Shift memory buffer for DIIS vectors
                self.__diis_vals[:-1] = ohcopy(self.__diis_vals[1:])
                self.__diis_err[:-1] = ohcopy(self.__diis_err[1:])

                # Store current iteration in DIIS buffer (Σ_HF + Σ_2B)
                self.__diis_vals[-1, 0] = ohcopy(self.sehf)
                self.__diis_vals[-1, 1:] = ohcopy(self.se2btau)

                # Error vector = ΔΣ between current and previous step
                self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]

                # Start extrapolation if enough vectors accumulated
                if iterations >= self.diis_mem:
                    fprint("Starting DIIS extrapolation", file=out_fl)

                    # Build error matrix B[i,j] = (error_i ⋅ error_j)
                    B = np.zeros((self.diis_mem,) * 2)
                    for i in range(self.diis_mem):
                        for j in range(i, self.diis_mem):
                            B[i, j] = np.sum((self.__diis_err[i] * self.__diis_err[j]).trace)
                            if i != j:
                                B[j, i] = B[i, j]
                    B /= np.mean(B)

                    # Solve Bc = 1 for DIIS weights
                    try:
                        Binv = np.linalg.inv(B)
                    except:
                        Binv = np.linalg.inv(B + np.eye(self.diis_mem) * 1e-8)  # regularization
                    c_prime = Binv @ np.ones((self.diis_mem,))
                    c = c_prime / np.sum(c_prime)

                    # Log weights
                    for k in range(self.diis_mem):
                        fprint(f"c{k} = {c[k]:.8f}", file=out_fl)

                    # Weighted sum of Σ vectors using DIIS coefficients
                    seext = ohsum(c[:, None] * self.__diis_vals, axis=0)
                    self.__sehf = seext[0]
                    self.__se2bl = ohfit(self.stauf, seext[1:])

                    # Save back to buffer
                    self.__diis_vals[-1] = ohcopy(seext)
                    self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]
            # =============== End DIIS block ===============

            # Update Green's function and phonon propagator for new Σ
            fprint("Computing gloc", file=out_fl)
            self.__update_green(out_fl)
            fprint("Computing phonon propagator", file=out_fl)
            self.__update_gb()
            fprint(f"Expected phononic excitations is {-2*np.sum(self.irbb.u(self.beta)*self.dl):.5f}", file=out_fl)
            fprint('\n', file=out_fl)

            # Check for convergence by computing change in Green's function
            iterations += 1
            conv = np.sum(((self.glocl - last_g)**2).sqrt().trace)
            self.__conv_ls.append(conv)

            fprint(f"iteration {iterations} finished with convergence {conv:.8e}", file=out_fl)
            fprint('-'*15, file=out_fl)
            fprint('\n'*2, file=out_fl)

            if conv <= tol:
                fprint("Finished", file=out_fl)
                self.__solved = True
                out_fl.close()
                return

            if iterations >= max_iter:
                fprint("Reached max iterations", file=out_fl)
                out_fl.close()
                return

            # Loop stagnation detection: check if convergence hasn't improved recently
            if iterations >= 5:
                loop_stuck = all(
                    abs(conv - self.conv_ls[-1 - ii]) <= tol / 1000
                    for ii in range(1, 5)
                )
                if loop_stuck:
                    if conv <= tol * 100:
                        fprint("Finished in a loop with satisfactory convergence", file=out_fl)
                        self.__solved = True
                    else:
                        fprint("Aborted in a loop", file=out_fl)
                        self.__solved = False
                    out_fl.close()
                    return

    def save(self, path):
        """
        Export all quantities to HDF5 using the `export_solver_to_hdf5` function.
        """
        if not self.__solved:
            print("Not solved yet, nothing to save")
            return
        export_solver_to_hdf5(self, path)
