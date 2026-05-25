# Figure Plotting Folder

This folder contains the data and plotting scripts for the paper "A regularised 
shadowing method for sensitivity analysis of turbulent flows" by Liang Fang and 
George Papadakis. The plotting scripts are grouped by problem

- `Figs_L96/`: Lorenz-96 figures.
- `Figs_Kol/`: Kolmogorov-flow figures.
- `Figs_CLV/`: covariant Lyapunov vector and angle figures.

Each folder has a `data/` directory. The scripts expect to be run from inside
their own figure folder, because the data paths use `./data/...`. The output PDF
names are fixed by the plotting scripts.

## How To Run

Run one script:

```bash
cd Figs_L96
python3 plot_fig3.py
```

Run all figure scripts from the root of this folder:

```bash
python3 run_figures.py
```

To run only one group, edit `FOLDERS_TO_RUN` at the top of `run_figures.py`, for
example:

```python
FOLDERS_TO_RUN = ("Figs_Kol",)
```

The runner executes each script with its working directory set to the matching
figure folder, so the existing `./data` paths and figure filenames are kept.

## Lorenz-96 Figures

| Script | Output figures | Data used |
| --- | --- | --- |
| `Figs_L96/plot_fig3.py` | `fig3a.pdf`, `fig3b.pdf` | `data/LE_information_L96_10_20.npz`, `data/LE_L96.dat` |
| `Figs_L96/plot_fig4_5.py` | `fig4a.pdf`, `fig4b.pdf`, `fig5.pdf` | `data/l_curve.npz` |
| `Figs_L96/plot_fig6.py` | `fig6_1.pdf`, `fig6_2.pdf` | `data/sol_15.0_29_0.npz`, `data/sol_15.0_29.npz` |
| `Figs_L96/plot_fig7_8.py` | `fig7a.pdf`, `fig7b.pdf`, `fig8a.pdf`, `fig8b.pdf` | `data/NILSS_L96_10.npz`, `data/NILSS_L96_regularised_10.npz`, `data/FD_derivative_J1_10.dat`, `data/FD_derivative_J2_10.dat` |
| `Figs_L96/plot_fig9.py` | `fig9a.pdf`, `fig9b.pdf` | `data/L96_conditions.npz` |
| `Figs_L96/plot_fig10.py` | `fig10a.pdf`, `fig10b.pdf` | `data/NILSS_L96_regularised_10.npz`, `data/NILSS_L96_regularised_10_2.npz`, `data/FD_derivative_J1_10.dat`, `data/FD_derivative_J2_10.dat` |
| `Figs_L96/plot_fig11.py` | `fig11a.pdf`, `fig11b.pdf` | `data/NILSS_L96_5.npz`, `data/NILSS_L96_regularised_5.npz`, `data/FD_derivative_J1_5.dat`, `data/FD_derivative_J2_5.dat` |


## Kolmogorov-Flow Figures

| Script | Output figures | Data used |
| --- | --- | --- |
| `Figs_Kol/plot_fig12.py` | `fig12a.pdf`, `fig12b.pdf` | `data/post_data_re_34.npz` |
| `Figs_Kol/plot_fig13.py` | `fig13a.pdf`, `fig13b.pdf` | `data/spectrum.npz` |
| `Figs_Kol/plot_fig14.py` | `fig14a.pdf`, `fig14b.pdf` | `data/LEs_Kol.npz` |
| `Figs_Kol/plot_fig15.py` | `fig15.pdf` | `data/Kol_38_energy.npz`, `data/Kol_38_energy_regularised.npz` |
| `Figs_Kol/plot_fig16.py` | `fig16a.pdf`, `fig16b.pdf`, `fig16c.pdf`, `fig16d.pdf` | `data/Kol_38_energy.npz`, `data/Kol_38_energy_regularised.npz`, `data/L_curve_r_14.npz`, `data/L_curve_r_28.npz` |
| `Figs_Kol/plot_fig17.py` | `fig17.pdf` | `data/NILSS_KolSol_1_34.0.npz`, `data/NILSS_KolSol_1_35.0.npz`, `data/NILSS_KolSol_1_36.0.npz`, `data/NILSS_KolSol_1_37.0.npz`, `data/NILSS_KolSol_1_38.0.npz` |
| `Figs_Kol/plot_fig18.py` | `fig18a.pdf`, `fig18b.pdf` | `data/Kol_conditions.npz` |


## CLV And Angle Figures

| Script | Output figures | Data used |
| --- | --- | --- |
| `Figs_CLV/plot_fig20.py` | `fig20_1.pdf` to `fig20_6.pdf` | `data/Kol_CLVs_contour.npz` |
| `Figs_CLV/plot_fig21.py` | `fig21a.pdf`, `fig21b.pdf`, `fig21c.pdf` | `data/Angles_Kol.npz` |
| `Figs_CLV/plot_fig22.py` | `fig22a.pdf`, `fig22b.pdf` | `data/L96_angles.npz` |
| `Figs_CLV/plot_fig23.py` | `fig23a.pdf`, `fig23b.pdf` | `data/KolSol_CLVs_23.npz`, `data/Angles_Kol.npz` |
| `Figs_CLV/plot_fig24.py` | `fig24a.pdf`, `fig24b.pdf` | `data/L96_CLVs_15.0_14.npz` |
| `Figs_CLV/plot_fig25.py` | `fig25a.pdf`, `fig25b.pdf` | `data/KolSol_CLVs_23.npz` |

## Data Inventory

The `.npz` files store NumPy arrays. Important groups are:

- NILSS derivative results: `NILSS_L96*.npz`, `NILSS_KolSol_1_*.npz`,
  `Kol_38_energy*.npz`.
- Finite-difference references: `FD_derivative_J*_*.dat`.
- Lyapunov spectra and positive-exponent counts: `LE_*.dat`,
  `LE_information_L96*.npz`, `LEs_Kol.npz`.
- Regularisation diagnostics: `l_curve.npz`, `L_curve_r_*.npz`,
  `L96_conditions.npz`, `Kol_conditions.npz`.
- Field/spectrum snapshots: `sol_15.0_29*.npz`, `post_data_re_34.npz`,
  `spectrum.npz`.
- CLV and angle data: `Kol_CLVs_contour.npz`, `KolSol_CLVs_23.npz`,
  `L96_CLVs_15.0_14.npz`, `Angles_Kol.npz`, `L96_angles.npz`.
