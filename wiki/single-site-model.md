# Single-Site Model: Theory and Implementation

This section describes the **single-site Jahn-Teller + spin-orbit + Coulomb interaction model**, along with its numerical implementation via exact diagonalization in Fock space.

---

## 🧠 Theoretical Background

We consider a single transition-metal site with partially filled t<sub>2g</sub> orbitals under the influence of:

- **Jahn-Teller (JT) distortions** in E<sub>g</sub> modes
- **Spin-orbit coupling (SOC)** (parameter λ)
- **Electron-electron interactions** via the Kanamori model

The total Hamiltonian is:

$$
H = H_{\text{JT}} + H_{\text{SOC}} + H_{\text{Kanamori}}
$$

---

### 🔹 Jahn-Teller Term

The JT coupling is defined in polar coordinates (Q, θ) via:

$$
H_{\text{JT}} = -Qg \left[\frac{1}{\sqrt{3}}(l_x^2 - l_y^2)\sin\theta + \left(l_z^2 - \frac{2}{3}\right)\cos\theta \right] + \frac{B}{2} Q^2
$$

Where:
- `Q` is the amplitude of distortion
- `g` is the coupling strength
- `B` is the elastic restoring coefficient

---

### 🔹 Spin-Orbit Coupling

SOC couples spin and orbital degrees of freedom:

$$
H_{\text{SOC}} = \frac{\lambda}{2} \, \mathbf{l} \cdot \boldsymbol{\sigma}
$$

It lifts the t<sub>2g</sub> degeneracy into:
- A lower **j = 3/2** quartet
- A higher **j = 1/2** doublet

---

### 🔹 Electron-Electron Interaction (Kanamori)

The Kanamori Hamiltonian includes:
- Intra-orbital repulsion $ U $
- Inter-orbital repulsion $ U' = U - 2J $
- Hund's coupling $ J $
- Pair-hopping and spin-flip exchange terms

This captures all Coulomb and exchange effects in the t<sub>2g</sub> shell.

---

## 🧮 Numerical Implementation

We solve the Hamiltonian using **exact diagonalization** on a Fock basis of 6 spin-orbitals:
- 3 t<sub>2g</sub> orbitals: `yz`, `zx`, `xy`
- Each with spin-↑ and spin-↓

Total modes: 6 → Fock space dimension: 64

---

### 🔧 Key Code Components

#### 🔹 Basis

```python
b = basisFock(6)  # generates Fock space of 6 modes
```

#### 🔹 Operators

```python
ayzu, ayzd, azxu, azxd, axyu, axyd = [annihilationFermi(k, b) for k in range(6)]
cyzu, cyzd, czxu, czxd, cxyu, cxyd = [creationFermi(k, b) for k in range(6)]
```

Used to define:
- Number operators: $ n_{\alpha\sigma} = c^\dagger_{\alpha\sigma} c_{\alpha\sigma} $
- Angular momentum: $ L_x, L_y, L_z $
- Spin operators: $ S_x, S_y, S_z $

---

### 🔹 Hamiltonians

- `HK(U, J)`: Kanamori interaction
- `Hsoc(lb)`: spin-orbit term (λ-dependent)
- `Hjt(Qx, Qy)`: JT term in cartesian Q-space

---

### 🔹 Energy Map Computation

Energy is computed over a grid in the (Qx, Qy) distortion space:

```python
emap[i, j] = eigobj([Qx[i, j], Qy[i, j]], N)
```

Where `eigobj(x, N)` extracts the **lowest eigenvalue** of:

$$
H_N(Q_x, Q_y) = H_{\text{JT}} + H_{\text{SOC}} + H_{\text{Kanamori}} + \frac{B}{2}(Q_x^2 + Q_y^2)
$$

Only a subset of the Fock space is used depending on electron count `N`.

---

## 💾 Output

For each combination of:
- Spin-orbit coupling `λ ∈ [0, 0.1, 0.3, 0.5]`
- Electron count `N ∈ {1, 2, 3, 4, 5}`

We save a text file with energy values on a (Qx, Qy) grid:

```plaintext
PaperFigs/T0data/{N}N_{int(10*lb)}SOC_lowee.txt
```

Each file contains:
```
Qx  Qy  E(Qx, Qy)
```

---

## 📈 Visualization



## 🧪 Results

This single-site model reveals:
- How SOC suppresses or reshapes Jahn-Teller distortions
- Which fillings (N) support static vs. dynamic JT behavior
- How low-energy orbital configurations shift with λ

It provides a foundational understanding before introducing intersite hybridization (see `LatticeModel.md`).
