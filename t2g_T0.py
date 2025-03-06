import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from itertools import combinations



def basisFock(num_modes):
    basis = []
    for N in range(num_modes+1):
        basis += list(combinations(range(num_modes), N))
    return basis


def annihilationFermi(mode, basis):
    size = len(basis)
    A = lil_matrix((size, size))
    for i, state in enumerate(basis):
        if mode in state:
            new_state = tuple(x for x in state if x != mode)
            sign = (-1) ** state.index(mode)  # Fermionic anti-commutation rule
            j = basis.index(new_state)
            A[j, i] = sign
    return A


def creationFermi(mode, basis):
    size = len(basis)
    C = lil_matrix((size, size))
    for i, state in enumerate(basis):
        if mode not in state:
            new_state = tuple(sorted(state + (mode,)))
            sign = (-1) ** sum(m < mode for m in state)  # Fermionic anti-commutation rule
            j = basis.index(new_state)
            C[j, i] = sign
    return C



b = basisFock(6)
ayzu, ayzd, azxu, azxd, axyu, axyd = [annihilationFermi(k, b) for k in range(6)]
cyzu, cyzd, czxu, czxd, cxyu, cxyd = [creationFermi(k, b) for k in range(6)]

nyzu = cyzu*ayzu; nyzd = cyzd*ayzd; nyz = nyzu + nyzd
nzxu = czxu*azxu; nzxd = czxd*azxd; nzx = nzxu + nzxd
nxyu = cxyu*axyu; nxyd = cxyd*axyd; nxy = nxyu + nxyd
Nop = nyz + nzx + nxy

lx = 1j*(czxu*axyu + czxd*axyd - cxyu*azxu - cxyd*azxd)
ly = 1j*(cxyu*ayzu + cxyd*ayzd - cyzu*axyu - cyzd*axyd)
lz = 1j*(cyzu*azxu + cyzd*azxd - czxu*ayzu - czxd*ayzd)

sx = 0.5*(cyzu*ayzd + cyzd*ayzu + czxu*azxd + czxd*azxu + cxyu*axyd + cxyd*axyu)
sy = -1j*0.5*(cyzu*ayzd - cyzd*ayzu + czxu*azxd - czxd*azxu + cxyu*axyd - cxyd*axyu)
sz = 0.5*(cyzu*ayzu - cyzd*ayzd + czxu*azxu - czxd*azxd + cxyu*axyu - cxyd*axyd)


U = 2.5
J = 0.2
HK = (U*(nyzu*nyzd + nzxu*nzxd + nxyu*nxyd) +
      (U-2*J)*(nyzu*nzxd + nyzu*nxyd + nzxu*nyzd + nzxu*nxyd + nxyu*nyzd + nxyu*nzxd) + 
      (U-3*J)*(nyzu*nzxu + nyzd*nzxd + nyzu*nxyu + nyzd*nxyd + nzxu*nxyu + nzxd*nxyd) -
      J*(cyzu*ayzd*czxd*azxu + cyzu*ayzd*cxyd*axyu + czxu*azxd*cyzd*ayzu + czxu*azxd*cxyd*axyu + cxyu*axyd*cyzd*ayzu + cxyu*axyd*czxd*azxu) +
      J*(cyzu*azxu*cyzd*azxd + cyzu*axyu*cyzd*axyd + czxu*ayzu*czxd*ayzd + czxu*axyu*czxd*axyd + cxyu*ayzu*cxyd*ayzd * cxyu*azxu*cxyd*azxd))


lb = 0.5
Hsoc = 0.5*lb * (cyzu*(1j*azxu - axyd) -
                 cyzd*(1j*azxd - axyu) -
                 czxu*(1j*ayzu - 1j*axyd) +
                 czxd*(1j*ayzd + 1j*axyu) +
                 cxyu*(ayzd - 1j*azxd) -
                 cxyd*(ayzu + 1j*azxu))


#%%
sl = {0:slice(0,1), 1:slice(1,7), 2:slice(7,22), 3:slice(22,42), 4:slice(42,57), 5:slice(57,63), 6:slice(63,64)}
g = 0.1
B = 0.1
size_grid = 101
Qmax = 1.2
for lb in [0,0.1,0.3,0.5,1]:
    print(lb)
    print('-'*5)
    Hsoc = 0.5*lb * (cyzu*(1j*azxu - axyd) -
                      cyzd*(1j*azxd - axyu) -
                      czxu*(1j*ayzu - 1j*axyd) +
                      czxd*(1j*ayzd + 1j*axyu) +
                      cxyu*(ayzd - 1j*azxd) -
                      cxyd*(ayzu + 1j*azxu))
    
    
    
    Hjt = lambda Q,th: Q*g * (np.sin(th)*(nyz-nzx)/np.sqrt(3) + np.cos(th)*(2*Nop/3 - nyz - nzx))
    
    
    def eigobj(x, N=1):
        sl = {0:slice(0,1), 1:slice(1,7), 2:slice(7,22), 3:slice(22,42), 4:slice(42,57), 5:slice(57,63), 6:slice(63,64)}
        HN = (HK + Hsoc + Hjt(x[0], x[1]))[sl[N],sl[N]]
        w = np.linalg.eigvalsh(HN.A)
        return w[0] + 0.5*B*x[0]**2
    
    
    Qx, Qy = np.meshgrid(*(np.linspace(-Qmax,Qmax,size_grid),)*2)
    for N in range(1,6):
        print(N)
        emap = np.zeros((size_grid,size_grid))
        Hjt = lambda qx,qy: g * (qy*(nyz-nzx)/np.sqrt(3) + qx*(2*Nop/3 - nyz - nzx))
        for i in range(size_grid):
            for j in range(size_grid):
                emap[i,j] = np.linalg.eigvalsh((HK + Hsoc + Hjt(Qx[i,j], Qy[i,j]))[sl[N],sl[N]].A)[0] + 0.5*B*(Qx[i,j]**2 + Qy[i,j]**2)
        
        emap -= np.min(emap)
        np.savetxt("PaperFigs/T0data/%iN_%iSOC_lowee.txt" % (N, 10*lb), np.array((Qx,Qy,emap)).reshape((3,emap.shape[0]*emap.shape[1])).T)
    
    print('-'*15+'\n')

fig, ax = plt.subplots()
plotable = ax.pcolormesh(Qx, Qy, emap, cmap='turbo', shading='gouraud')#, vmax = np.min(emap) + (np.max(emap)-np.min(emap))*0.15)
ax.set_xlabel(r"$Q^\theta$", fontsize=12)
ax.set_ylabel(r"$Q^\phi$", fontsize=12)
ax.set_title(r"$N=%i$" % N)
ax.set_aspect("equal")
clb = plt.colorbar(plotable)
clb.ax.set_title("Energy (eV)")