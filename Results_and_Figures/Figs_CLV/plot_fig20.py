# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
import os
from scipy.spatial.distance import cosine

params = {'legend.fontsize': 7,
          'figure.figsize': (4, 3),
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


Lyapunov = np.load(f"./data/Kol_CLVs_contour.npz")
CLVs = Lyapunov["CLVs"]
u = Lyapunov["U0"]
nx, n_le = CLVs.shape
CLVu = CLVs[:,:-2].copy()
CLVn = CLVs[:,n_le-2:n_le].copy()
ff = Lyapunov["F0"][0]

resolution = 32
x = y = np.arange(0, 2*np.pi, 2*np.pi/resolution)
X,Y = np.meshgrid(x,y)
extent = np.min(x), 2*np.pi, np.min(y), 2*np.pi

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0, 4.0), dpi=300, gridspec_kw={'width_ratios': [1, 0.05]})
ax1 = axes[0]
im1 = ax1.imshow(np.transpose(ff[...,0]),interpolation='spline16', cmap=plt.get_cmap('RdBu_r'),aspect=1,extent=extent,origin='lower')
ax1.set_ylabel(r'$y$',fontsize=12)
ax1.set_xlabel(r'$x$',fontsize=12)
cb_ax = axes[1]
cb = fig.colorbar(im1, cax=cb_ax)
plt.savefig(f"fig20_6.pdf")

for i in range(n_le):
    cv = np.reshape(CLVs[:,i],(32,32,2))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0, 4.0), dpi=300, gridspec_kw={'width_ratios': [1, 0.05]})
    ax1 = axes[0]
    im1 = ax1.imshow(np.transpose(cv[:,:,0]),interpolation='spline16', cmap=plt.get_cmap('RdBu_r'),aspect=1,extent=extent,origin='lower')
    ax1.set_ylabel(r'$y$',fontsize=12)
    ax1.set_xlabel(r'$x$',fontsize=12)
    cb_ax = axes[1]
    cb = fig.colorbar(im1, cax=cb_ax)
    plt.savefig(f"fig20_{i+1}.pdf")
