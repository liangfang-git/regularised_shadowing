import os
import numpy as np
from scipy.stats import norm
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'figure.figsize': (3, 3),
          'lines.linewidth': 1,
          'figure.dpi': 300,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize':10,
         'ytick.labelsize':10,
         'figure.autolayout':True}
pylab.rcParams['mathtext.fontset'] = 'stix'
pylab.rcParams['font.family'] = 'DeJavu Serif'
pylab.rcParams['font.serif'] = ['Times New Roman']
pylab.rcParams.update(params)

resolution = 32
x = y = np.arange(0, 2*np.pi, 2*np.pi/resolution)
X,Y = np.meshgrid(x,y)
extent = np.min(x), 2*np.pi, np.min(y), 2*np.pi

def plot_and_save_contour(u,name,colormap):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0, 4.0), dpi=300, gridspec_kw={'width_ratios': [1, 0.05]})
    ax1 = axes[0]
    im1 = ax1.imshow(np.transpose(u),interpolation='spline16', cmap=plt.get_cmap(colormap),aspect=1,extent=extent,origin='lower')
    ax1.set_ylabel(r'$y$',fontsize=12)
    ax1.set_xlabel(r'$x$',fontsize=12)
    cb_ax = axes[1]
    cb = fig.colorbar(im1, cax=cb_ax)
    figure_name = name+".pdf"
    plt.savefig(figure_name,format='pdf')

D = np.load(f"./data/post_data_re_34.npz")
meanvor = D["meanvor"]
extreme_vorticity_energy = D["extreme_vorticity_energy"]

plot_and_save_contour(meanvor,"fig12b",'RdBu_r')
plot_and_save_contour(extreme_vorticity_energy,"fig12a",'RdBu_r')

