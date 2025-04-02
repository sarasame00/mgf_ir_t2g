import numpy as np
import sparse_ir as ir

from maths.ohmatrix import ohmatrix, ohsum, ohcopy, ohzeros
from maths.utils import fprint, ohfit, ohevaluate
from simulation import self_energy
from simulation.export import export_solver_to_hdf5


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
        Self-consistently adjust chemical potential μ to match target density N.
        Uses fixed-point iteration with adaptive step size and logging.
        """
        self.__mu = np.sum((self.sehf + ohsum(self.irbf.u(self.beta)*(self.se2bl+2*self.seepl))).real.eigvals)/2
        last_sign = 0
        while True:
            fprint("Starting with mu=%.8f" % self.mu, out_fl)
            if self.__t != 0:
                gkiw = (
                    self.freqf[:,None,None,None] - self.Hlatt[None,:,:,:]
                    - self.sehf - self.sephm[None,:,:,:] - 2*self.seepiw[:,None,None,None]
                    + self.mu - 0.5*self.lbd*ohmatrix(0,1)
                )**-1
                self.__gkl = ohfit(self.smatf, gkiw, axis=0).real
                glociw = ohsum(gkiw, axis=(1,2,3)) / self.k_sz**3
            else:
                glociw = (
                    self.freqf - self.sehf - 2*self.seepiw
                    + self.mu - 0.5*self.lbd*ohmatrix(0,1)
                )**-1
            self.__glocl = ohfit(self.smatf, glociw).real
            Nexp = -6 * np.sum(self.irbf.u(self.beta) * self.glocl.a)
            fprint("Finished with Nexp=%.8f" % (Nexp.real), out_fl)
            DN = self.N - Nexp
            if abs(DN) <= tol:
                return
            if DN > 0:
                if last_sign == -1: delta /= 2
                self.__mu += delta
                last_sign = +1
            elif DN < 0:
                if last_sign == +1: delta /= 2
                self.__mu -= delta
                last_sign = -1

    def __update_gb(self):
        """
        Updates the phonon propagator D(iω) using bare D₀ and self-energy Σ_b.
        """
        d0iw = (2 * self.w0 / (self.freqb**2 - self.w0**2)).real
        diw = (d0iw**(-1) - self.sebiw)**(-1)
        self.__dl = self.smatb.fit(diw).real

    # ------------- Main solver loop -------------

    def solve(self, diis_active=True, tol=1e-6, max_iter=10000):
        """
        Run full self-consistent Dyson iteration with optional DIIS acceleration.
        """
        if self.diis_mem == 0:
            diis_active = False

        out_fl = open(self.__fl, 'w')
        fprint("Starting execution with the following parameters", file=out_fl)
        for name in ["T", "beta", "wM", "N", "t", "U", "J", "Jphm", "w0", "g", "lbd", "k_sz", "diis_mem"]:
            fprint(f"{name}={getattr(self, name):.3f}", file=out_fl)
        fprint("-" * 15 + "\n", file=out_fl)

        # Initial G, D update
        fprint("Computing non-interacting Green's function", file=out_fl)
        self.__update_green(out_fl)
        self.__update_gb()
        fprint('\n'*2, file=out_fl)

        iterations = 0
        while True:
            last_g = ohcopy(self.glocl)
            fprint(f"Starting iteration {iterations + 1}", file=out_fl)
            fprint("Updating self-energies", file=out_fl)

            # Compute all components of Σ
            self_energy.update_sehf(self)
            self_energy.update_sephm(self)
            self_energy.update_se2b(self)
            self_energy.update_seep(self)
            self_energy.update_seb(self)

            # DIIS acceleration (if enabled)
            if diis_active:
                self.__diis_vals[:-1] = ohcopy(self.__diis_vals[1:])
                self.__diis_err[:-1] = ohcopy(self.__diis_err[1:])
                self.__diis_vals[-1,0] = ohcopy(self.sehf)
                self.__diis_vals[-1,1:] = ohcopy(self.se2btau)
                self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]

                if iterations >= self.diis_mem:
                    fprint("Starting DIIS extrapolation", file=out_fl)
                    B = np.zeros((self.diis_mem,) * 2)
                    for i in range(self.diis_mem):
                        for j in range(i, self.diis_mem):
                            B[i, j] = np.sum((self.__diis_err[i] * self.__diis_err[j]).trace)
                            if i != j: B[j, i] = B[i, j]
                    B /= np.mean(B)
                    try:
                        Binv = np.linalg.inv(B)
                    except:
                        Binv = np.linalg.inv(B + np.eye(self.diis_mem)*1e-8)
                    c_prime = Binv @ np.ones((self.diis_mem,))
                    c = c_prime / np.sum(c_prime)
                    for k in range(self.diis_mem):
                        fprint(f"c{k} = {c[k]:.8f}", file=out_fl)
                    seext = ohsum(c[:, None] * self.__diis_vals, axis=0)
                    self.__sehf = seext[0]
                    self.__se2bl = ohfit(self.stauf, seext[1:])
                    self.__diis_vals[-1] = ohcopy(seext)
                    self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]

            fprint("Computing gloc", file=out_fl)
            self.__update_green(out_fl)
            fprint("Computing phonon propagator", file=out_fl)
            self.__update_gb()
            fprint(f"Expected phononic excitations is {-2*np.sum(self.irbb.u(self.beta)*self.dl):.5f}", file=out_fl)
            fprint('\n', file=out_fl)

            # Convergence check
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

            # Check for convergence stagnation
            if iterations >= 5:
                loop_stuck = all(
                    abs(conv - self.conv_ls[-1 - ii]) <= tol/1000
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
