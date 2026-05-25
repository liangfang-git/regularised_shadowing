import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'figure.figsize': (4, 3),
          'lines.linewidth': 1,
          'figure.dpi': 300,
         'axes.labelsize': 12,
         'axes.titlesize': 12,
         'xtick.labelsize':9,
         'ytick.labelsize':9,
         'figure.autolayout':True}
pylab.rcParams['mathtext.fontset'] = 'stix'
pylab.rcParams['font.family'] = 'DeJavu Serif'
pylab.rcParams['font.serif'] = ['Times New Roman']
pylab.rcParams.update(params)


nr = 10
n_lyap = 40
c_lb = 10
c_ub = 20
c_dist = 1
c_nb = int((c_ub - c_lb) / c_dist + 1)
c_arr = np.round(np.linspace(c_lb, c_ub, c_nb), 1)

LEs = np.load(f"./data/LE_information_L96_10_20.npz")
nple = LEs["nple"]
leading_le = LEs["leading_le"]
les = LEs["les"]


fig, ax = plt.subplots(figsize=(4,3)) 
ax.axhline(0.0, linestyle='-', linewidth=1.0, color='k')
le0 = np.loadtxt("./data/LE_L96.dat")
x = np.linspace(1,n_lyap,n_lyap)
ax.plot(x, les[:, 0], '-o', markersize=1.5, clip_on=True,
        linewidth=1.1, label=r'$F=10$')
ax.plot(x, les[:, 2], '-o', markersize=1.5, clip_on=True,
        linewidth=1.1, label=r'$F=12$')
ax.plot(x, les[:, 4], '-o', markersize=1.5, clip_on=True,
        linewidth=1.1, label=r'$F=14$')
ax.plot(x, les[:, 6], '-o', markersize=1.5, clip_on=True,
        linewidth=1.1, label=r'$F=16$')
ax.plot(x, les[:, 8], '-o', markersize=1.5, clip_on=True,
        linewidth=1.1, label=r'$F=18$')
ax.plot(x, les[:,10], '-o', markersize=1.5, clip_on=True,
        linewidth=1.1, label=r'$F=20$')

for i in range(len(le0[:25, 0])):
    ax.text(le0[i, 0], le0[i, 1], r'$\mathbf{+}$',
            fontsize=8, ha='center', va='center',
            color='k', clip_on=True)

ax.set_xlabel(r'index($i$)')
ax.set_ylabel(r'$\lambda_i$')
ax.grid(color='grey', linestyle='--', linewidth=0.3)

x_min, x_max = 0.5, 25.5
y_min, y_max = -2.0, 5.0

pad_x = 0.02 * (x_max - x_min)   
pad_y = 0.02 * (y_max - y_min)

ax.set_xlim(x_min - pad_x, x_max + pad_x)
ax.set_ylim(y_min - pad_y, y_max + pad_y)

ax.set_xticks(np.arange(1, 26, step=4))
ax.legend()

for artist in ax.get_children():
    if hasattr(artist, "set_clip_path") and artist is not ax.patch:
        artist.set_clip_path(ax.patch)

for gl in ax.get_xgridlines() + ax.get_ygridlines():
    gl.set_clip_on(True)
    gl.set_clip_path(ax.patch)

fig.tight_layout()
fig.savefig('./fig3a.pdf', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(4, 3))
bars = ax.bar(c_arr, nple, color='C0', width=0.65, alpha=0.8)    
for b in bars:
    b.set_clip_on(True)
    b.set_clip_path(ax.patch)
ax.set_xlabel(r'$F$')
ax.set_ylabel(r'$n_p$')
ymin, ymax = ax.get_ylim()
pad_y = 0.03 * (ymax - ymin)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
fig.tight_layout()
fig.savefig('./fig3b.pdf', bbox_inches='tight')
