import numpy as np
from scipy.sparse import lil_matrix
from itertools import combinations

class SingleSiteHamiltonian:
    def __init__(self, num_modes=6):
        self.num_modes = num_modes
        self.basis = self._build_fock_basis()
        self._init_operators()

    def _build_fock_basis(self):
        basis = []
        for N in range(self.num_modes + 1):
            basis += list(combinations(range(self.num_modes), N))
        return basis

    def _annihilation_operator(self, mode):
        A = lil_matrix((len(self.basis), len(self.basis)))
        for i, state in enumerate(self.basis):
            if mode in state:
                new_state = tuple(x for x in state if x != mode)
                sign = (-1) ** state.index(mode)
                j = self.basis.index(new_state)
                A[j, i] = sign
        return A

    def _creation_operator(self, mode):
        C = lil_matrix((len(self.basis), len(self.basis)))
        for i, state in enumerate(self.basis):
            if mode not in state:
                new_state = tuple(sorted(state + (mode,)))
                sign = (-1) ** sum(m < mode for m in state)
                j = self.basis.index(new_state)
                C[j, i] = sign
        return C

    def _init_operators(self):
        b = self.basis
        a = [self._annihilation_operator(k) for k in range(6)]
        c = [self._creation_operator(k) for k in range(6)]
        self.annihilators = a
        self.creators = c

        self.nyz = c[0] * a[0] + c[1] * a[1]
        self.nzx = c[2] * a[2] + c[3] * a[3]
        self.nxy = c[4] * a[4] + c[5] * a[5]
        self.Nop = self.nyz + self.nzx + self.nxy

        # Store for later use in interactions
        self._define_spin_and_orbital_operators()

    def _define_spin_and_orbital_operators(self):
        c, a = self.creators, self.annihilators
        self.lx = 1j * (c[2]*a[4] + c[3]*a[5] - c[4]*a[2] - c[5]*a[3])
        self.ly = 1j * (c[4]*a[0] + c[5]*a[1] - c[0]*a[4] - c[1]*a[5])
        self.lz = 1j * (c[0]*a[2] + c[1]*a[3] - c[2]*a[0] - c[3]*a[1])
        self.sx = 0.5*(c[0]*a[1] + c[1]*a[0] + c[2]*a[3] + c[3]*a[2] + c[4]*a[5] + c[5]*a[4])
        self.sy = -1j*0.5*(c[0]*a[1] - c[1]*a[0] + c[2]*a[3] - c[3]*a[2] + c[4]*a[5] - c[5]*a[4])
        self.sz = 0.5*(c[0]*a[0] - c[1]*a[1] + c[2]*a[2] - c[3]*a[3] + c[4]*a[4] - c[5]*a[5])

    def HK(self, U, J):
        c, a = self.creators, self.annihilators
        nyzu, nyzd = c[0]*a[0], c[1]*a[1]
        nzxu, nzxd = c[2]*a[2], c[3]*a[3]
        nxyu, nxyd = c[4]*a[4], c[5]*a[5]
        nyz = nyzu + nyzd
        nzx = nzxu + nzxd
        nxy = nxyu + nxyd

        return (
            U * (nyzu*nyzd + nzxu*nzxd + nxyu*nxyd)
            + (U - 2*J) * (nyzu*nzxd + nyzu*nxyd + nzxu*nyzd + nzxu*nxyd + nxyu*nyzd + nxyu*nzxd)
            + (U - 3*J) * (nyzu*nzxu + nyzd*nzxd + nyzu*nxyu + nyzd*nxyd + nzxu*nxyu + nzxd*nxyd)
        )

    def Hsoc(self, lb):
        c, a = self.creators, self.annihilators
        return 0.5 * lb * (
            c[0]*(1j*a[2] - a[5]) - c[1]*(1j*a[3] - a[4])
            - c[2]*(1j*a[0] - 1j*a[5]) + c[3]*(1j*a[1] + 1j*a[4])
            + c[4]*(a[1] - 1j*a[3]) - c[5]*(a[0] + 1j*a[2])
        )

    def Hjt(self, g, qx, qy):
        return g * (qy*(self.nyz - self.nzx)/np.sqrt(3) + qx*(2*self.Nop/3 - self.nyz - self.nzx))

    def eigval(self, x, U, J, lb, N, B, g):
        sl = {
            0: slice(0, 1), 1: slice(1, 7), 2: slice(7, 22), 3: slice(22, 42),
            4: slice(42, 57), 5: slice(57, 63), 6: slice(63, 64)
        }
        H = self.HK(U, J) + self.Hsoc(lb) + self.Hjt(g, x[0], x[1])
        HN = H[sl[N], sl[N]]
        w = np.linalg.eigvalsh(HN.toarray())
        return w[0] + 0.5 * B * (x[0]**2 + x[1]**2)

def run_single_site(x, U, J, lb, N, B, g):
    h = SingleSiteHamiltonian()
    eigval = h.eigval(x, U, J, lb, N, B, g)
    return {
        "eigenvalue": eigval,
        "meta": {
            "x": x, "U": U, "J": J, "lb": lb, "N": N, "B": B, "g": g
        }
    }
