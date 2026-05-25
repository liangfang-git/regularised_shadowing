# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
params = {'legend.fontsize': 7,
          'figure.figsize': (5, 3),
          'lines.linewidth': 1,
          'figure.dpi': 300,
         'axes.labelsize': 8,
         'axes.titlesize': 8,
         'xtick.labelsize':8,
         'ytick.labelsize':8,
         'figure.autolayout':True}
pylab.rcParams['mathtext.fontset'] = 'stix'
pylab.rcParams['font.family'] = 'DeJavu Serif'
pylab.rcParams['font.serif'] = ['Times New Roman']
pylab.rcParams.update(params)

Sol0 = np.load("./data/Kol_38_energy.npz")
t = Sol0['t']
dJ0 = Sol0['dJ']
m0 = np.mean(dJ0,axis=-1)
st0 = np.array(dJ0).std(1,ddof=1)

Sol1 = np.load("./data/Kol_38_energy_regularised.npz")
dJ1 = Sol1['dJ']
m1 = np.mean(dJ1,axis=-1)
st1 = np.array(dJ1).std(1,ddof=1)
color0 = "C0"
color1 = "C3"

i = 14
L = np.load(f"./data/L_curve_r_{i}.npz")
rho = L["rho"]
eta = L["eta"]
ii = L["max_curvature_index"]
rho_c = rho[ii]
eta_c = eta[ii]
plt.figure(figsize=(3.5,3))
plt.plot(rho,eta,'-',markersize=2,lw=1)
plt.plot(rho_c,eta_c,'o')
plt.xscale("log",base=10)
plt.yscale("log",base=10)
plt.xlabel(r"$\eta(\gamma)$")
plt.ylabel(r"$\rho(\gamma)$")
plt.grid()
plt.savefig(f"fig16a.pdf",format='pdf')

plt.figure(figsize=(3.5,3))
plt.plot(t, dJ0[:,i], '-o', markersize=2, markeredgecolor=color0, markerfacecolor='white',color = color0,label='Standard NILSS')
plt.plot(t, dJ1[:,i], '--o', markersize=2, markeredgecolor=color1, markerfacecolor='white',color = color1,label='Regularised NILSS')
plt.plot([t[0],t[-1]], [0.0161,0.0161], '--', color = 'k')
plt.ylim([0.007,0.027])
plt.xlabel(r'$T$')
plt.ylabel(r'$\frac{d\overline{J}}{d Re}$')
plt.legend()
plt.grid(True)
plt.savefig(f"fig16c.pdf",format = 'pdf')


i = 28
L = np.load(f"./data/L_curve_r_{i}.npz")
rho = L["rho"]
eta = L["eta"]
ii = L["max_curvature_index"]
rho_c = rho[ii]
eta_c = eta[ii]
plt.figure(figsize=(3.5,3))
plt.plot(rho,eta,'-',markersize=2,lw=1)
plt.plot(rho_c,eta_c,'o')
plt.xscale("log",base=10)
plt.yscale("log",base=10)
plt.xlabel(r"$\eta(\gamma)$")
plt.ylabel(r"$\rho(\gamma)$")
plt.grid()
plt.savefig(f"fig16b.pdf",format='pdf')

plt.figure(figsize=(3.5,3))
plt.plot(t, dJ0[:,i], '-o', markersize=2, markeredgecolor=color0, markerfacecolor='white',color = color0,label='Standard NILSS')
plt.plot(t, dJ1[:,i], '--o', markersize=2, markeredgecolor=color1, markerfacecolor='white',color = color1,label='Regularised NILSS')
plt.plot([t[0],t[-1]], [0.0161,0.0161], '--', color = 'k')
plt.ylim([0.007,0.027])
plt.xlabel(r'$T$')
plt.ylabel(r'$\frac{d\overline{J}}{d Re}$')
plt.legend()
plt.grid(True)
plt.savefig(f"fig16d.pdf",format = 'pdf')
