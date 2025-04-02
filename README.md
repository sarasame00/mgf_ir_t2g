## 📁 Project Structure

```plaintext
MGR_IR_T2G/
├── single_site/                    # Reference calculations: single-site JT model
│   ├── single-site.py              # JT Hamiltonian: t ⊗ E problem, orbital energy surfaces
│   ├── single-site-plots.py 
│   ├── data/                   
│   └── figures/                    # Output figures from single-site simulations
│
└──t2g_jt_soc/
    ├── main.py
    │   └─ 🚀 Entry point for simulations: loads parameters, runs solver, logs output
    │
    ├── config/
    │   └── parameters.py       # Defines sweepable parameters (U, J, ξ, g, t, N, etc.)
    │
    ├── data/
    │   ├── logs/                   # Logs from simulation (DIIS, energy, debug info)
    │   ├── results/                # Output: G(ω), Σ(ω), correlators, etc.
    │   └── simulated_values.csv    # Summary table of runs (N, t, ξ → energy, JT gain, etc.)
    │
    ├── maths/
    │   ├── __init__.py
    │   ├── ohmatrix.py             # Class `OhMatrix`: M = a·I + b·V matrix algebra
    │   └── utils.py                # ohfit, ohevaluate, fprint, matsubara_frequencies
    │
    ├── simulation/
    │   ├── __init__.py
    │   ├── runner.py               # Orchestrates multiple parameter runs
    │   ├── dyson_solver.py         # `DysonSolver`: builds G₀⁻¹ and solves Dyson equations
    │   ├── export.py               
    │   └── self_energy.py          # Σ(ω): Hartree-Fock, Born, phonon, orbital exchange
    │
    ├── analysis/                   # Tools to analyze and visualize results
    │   ├── visualize.py
    │   └── loaders.py
    │    
    ├── README.md                   # Project overview, theory, and usage instructions
    └── requirements.txt            # Python dependencies (numpy, scipy, sparse_ir, etc.)
    