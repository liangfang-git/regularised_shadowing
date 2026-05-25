# use single direction and N time segments to solve Lorenz problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import welch
import re
import matplotlib.pyplot as plt

# -----------------------
# Plot style
# -----------------------
params = {'legend.fontsize': 9,
          'figure.figsize': (4,3.5),
          'lines.linewidth': 1.2,
          'figure.dpi': 300,
          'axes.labelsize': 8,
          'axes.titlesize': 9,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'figure.autolayout': True}
pylab.rcParams.update(params)
pylab.rcParams['mathtext.fontset'] = 'stix'
pylab.rcParams['font.family'] = 'DeJavu Serif'
pylab.rcParams['font.serif'] = ['Times New Roman']

# -----------------------
# Load data
# -----------------------
S0 = np.load("./data/sol_15.0_29_0.npz")
S  = np.load("./data/sol_15.0_29.npz")

v0 = S0["v_out"]   # (nt, nx)
v  = S["v_out"]

nt, nx = v.shape
x_i = np.arange(1, nx + 1)

# -----------------------
# Time settings
# -----------------------
dt = 0.01
fs = 1.0 / dt              # 100
t_v = np.arange(nt) * dt   # 0..40 (since nt=4001)

# -----------------------
# Helpers: plotting fields
# -----------------------
def robust_sym_limits(M, q=0.995):
    a = np.nanquantile(np.abs(M), q)
    return -a, a

def plot_field_independent(M, t, x, title, index, cmap="RdBu_r", use_percentile=True, q=0.995):
    extent = [x[0], x[-1], t[-1], t[0]]
    if use_percentile:
        vmin, vmax = robust_sym_limits(M, q=q)
    else:
        vmax = np.nanmax(np.abs(M))
        vmin, vmax = -vmax, vmax

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    plt.figure()
    im = plt.imshow(M, aspect="auto", origin="upper", extent=extent, cmap=cmap, norm=norm)
    plt.xlabel(r"$i$")
    plt.ylabel(r"$t$")
    plt.title(title)
    cb = plt.colorbar(im)
    cb.ax.tick_params(labelsize=7)
    plt.savefig(f"fig6_{index}.pdf", dpi=300)


# -----------------------
# 1) Field plots (independent color ranges)
# -----------------------
plot_field_independent(v0, t_v, x_i, r"$v_s$ before regularisation", 1, use_percentile=True, q=0.997)
plot_field_independent(v,  t_v, x_i, r"$v_s$ after regularisation", 2, use_percentile=True, q=0.997)


