import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 7,
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

k = np.linspace(1,16,16)
S = np.load("./data/spectrum.npz")

plt.figure()
plt.plot(k,np.mean(S["sx"],axis=-1),'o-',linewidth=1., markersize=3)
plt.yscale('log')
plt.xlabel(r'$k_x$')
plt.xticks(np.arange(1, 16, step=2))
plt.ylabel('Turbulence kinetic energy')
plt.grid(color='grey', linestyle='--', linewidth=0.3)
plt.savefig('fig13a.pdf')

plt.figure()
plt.plot(k,np.mean(S["sy"],axis=-1),'o-',linewidth=1., markersize=3)
plt.yscale('log')
plt.xlabel(r'$k_y$')
plt.xticks(np.arange(1, 16, step=2))
plt.ylabel('Turbulence kinetic energy')
plt.grid(color='grey', linestyle='--', linewidth=0.3)
plt.savefig('fig13b.pdf')
