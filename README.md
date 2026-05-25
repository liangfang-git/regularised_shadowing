# Regularised Shadowing Data and Code

This folder contains the code, input data, processed results, and plotting
scripts for the paper "A regularised shadowing method for sensitivity analysis
of turbulent flows" by Liang Fang and George Papadakis.

The repository focuses on two chaotic dynamical systems:

- Lorenz-96 (`L96`)
- Kolmogorov flow (`Kol`)

and includes workflows for Lyapunov exponent/covariant Lyapunov vector
calculations, NILSS sensitivity analysis, regularised NILSS sensitivity
analysis, and figure reproduction.

## Folder Layout

- `Codes/`: simulation and analysis code.
  - `Lyapunov/L96/`: Lorenz-96 Lyapunov and CLV calculations.
  - `Lyapunov/Kol/`: Kolmogorov-flow Lyapunov and CLV calculations.
  - `Regularised_Shadowing/L96/`: Lorenz-96 NILSS and regularised NILSS runs.
  - `Regularised_Shadowing/Kol/`: Kolmogorov-flow NILSS and regularised NILSS runs.
- `Results_and_Figures/`: processed data and scripts used to reproduce the
  paper figures.
  - `Figs_L96/`: Lorenz-96 figures.
  - `Figs_Kol/`: Kolmogorov-flow figures.
  - `Figs_CLV/`: CLV and angle figures.

More detailed notes are available in:

- `Codes/README.md`
- `Results_and_Figures/README.md`

## Requirements

Use Python 3. The scripts use standard scientific Python packages:

```bash
python3 -m pip install numpy scipy matplotlib pandas seaborn statsmodels einops opt-einsum
```

The `.npz` files are NumPy archives, `.dat` files are text data, and `.pkl`
files contain stored initial conditions for Kolmogorov-flow runs.

## Reproduce Figures

To regenerate all paper figures from the processed data:

```bash
cd Results_and_Figures
python3 run_figures.py
```

To run a single figure script, run it from its own figure folder, for example:

```bash
cd Results_and_Figures/Figs_L96
python3 plot_fig3.py
```

The plotting scripts write PDF files beside the script being run.

## Run Simulation Workflows

The main driver scripts are:

```bash
cd Codes/Lyapunov/L96
python3 L96_Lyapunov.py

cd Codes/Lyapunov/Kol
python3 Kol_Lyapunov.py

cd Codes/Regularised_Shadowing/L96
python3 L96_NILSS.py

cd Codes/Regularised_Shadowing/Kol
python3 Kol_NILSS.py
```

Run these scripts from their own directories because they use relative paths
such as `./data/...`. Simulation settings are defined near the top of each
driver script. These runs can be computationally expensive.

## Outputs

Simulation outputs are written beside the relevant driver script, commonly as
`.npz` files. Figure outputs are written as `.pdf` files beside the plotting
scripts.

The included `Results_and_Figures/data` folders contain processed data that can
be used directly for figure reproduction without rerunning the full simulations.
