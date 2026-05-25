# Lyapunov and Regularised Shadowing Codes

This repository contains Python scripts and input data for chaotic dynamical
systems experiments. It covers Lyapunov exponent/covariant Lyapunov vector
calculations and NILSS sensitivity analysis for:

- Lorenz-96 (`L96`)
- Kolmogorov flow (`Kol`)

The code is organised by method first, then by dynamical system.

## Layout

- `Lyapunov/L96/`
  - `L96_Lyapunov.py`: computes Lorenz-96 CLVs for forcing values `F = 10, ..., 20`
  - `L96_functions.py`: Lorenz-96 model and tangent equations
  - `Lyapunov_functions.py`: Lyapunov/CLV routines
  - `data/L96_initials_10-20.npz`: initial conditions

- `Lyapunov/Kol/`
  - `Kol_Lyapunov.py`: computes Kolmogorov-flow CLVs at `Re = 38`
  - `KolSol_functions.py`: Kolmogorov-flow model
  - `Lyapunov_functions.py`: Lyapunov/CLV routines
  - `initials/*.pkl`: input initial conditions
  - `w_random.npz`: random tangent directions

- `Regularised_Shadowing/L96/`
  - `L96_NILSS.py`: computes Lorenz-96 sensitivities with regularised NILSS
  - `NILSS.py`: NILSS implementation
  - `L96_functions.py`: Lorenz-96 model, tangent equations, and objectives
  - `data/*.npz`: initial conditions, random directions, and Lyapunov information

- `Regularised_Shadowing/Kol/`
  - `Kol_NILSS.py`: computes Kolmogorov-flow sensitivities with regularised NILSS
  - `NILSS.py`: NILSS implementation
  - `KolSol_functions.py`: Kolmogorov-flow model and objectives
  - `data/`: Lyapunov information, random directions, and initial conditions

## Requirements

Use Python 3 with the following packages:

```bash
python3 -m pip install numpy scipy matplotlib einops opt-einsum
```

`matplotlib` is used by the NILSS routines when l-curve plotting is enabled.

## Running

Run each script from its own directory. The scripts use relative paths such as
`./data/...` and write outputs beside the script unless noted otherwise.

### Lorenz-96 Lyapunov/CLV

```bash
cd Lyapunov/L96
python3 L96_Lyapunov.py
```

This uses `data/L96_initials_10-20.npz`, loops over `F = 10, ..., 20` and
`nr = 50` repeats, and writes files named:

```text
NILSS_L96_CLVs_<F>_<repeat>.npz
```

### Lorenz-96 Regularised NILSS

```bash
cd Regularised_Shadowing/L96
python3 L96_NILSS.py
```

This computes sensitivities of:

- `J1`: spatial-temporal mean state
- `J2`: spatial-temporal mean energy

The main output is:

```text
NILSS_L96_regularised.npz
```

It contains `c_arr`, `Jr`, `dJr`, `T_seg`, and `T_control`.

### Kolmogorov-Flow Lyapunov/CLV

```bash
cd Lyapunov/Kol
python3 Kol_Lyapunov.py
```

This uses the included `Re = 38` initial conditions in `initials/` and writes:

```text
results/NILSS_KolSol_CLVs_<repeat>.npz
new_initials/Kol_38_<repeat>.npz
```

### Kolmogorov-Flow Regularised NILSS

```bash
cd Regularised_Shadowing/Kol
python3 Kol_NILSS.py
```

This loops over `Re = 34, ..., 38` and computes sensitivities of:

- spatial-temporal mean energy
- spatial-temporal mean enstrophy

It writes one file per Reynolds number:

```text
NILSS_KolSol_<regularised>_<Re>.npz
```

and the combined output:

```text
NILSS_KolSol_regularised.npz
```

## Configuration

Simulation settings are defined near the top of each driver script. Common
settings include the time step, system size, control parameter range, number of
repeats, total control time, and whether regularisation or l-curve plotting is
enabled.

Edit those config blocks directly for a different experiment.
