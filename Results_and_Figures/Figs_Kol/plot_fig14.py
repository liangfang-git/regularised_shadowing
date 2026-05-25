import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 9,
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


LE = np.load("./data/LEs_Kol.npz")
c_arr = LE["c_arr"]
les = LE["les"]
leading_le = LE["leading_le"]
nple = LE["nple"]

plt.figure()
plt.plot(c_arr, np.zeros_like(c_arr),'k-')
for i in range(5):
    plt.plot(c_arr, les[i,:], '-o',markerfacecolor='1',markersize=3.2, linewidth=1.5,label=rf'$\lambda_{i+1}$')
plt.xlabel(r'$Re$')
plt.ylabel(r'$\lambda_i$')
plt.ylim([-0.01,0.13])
plt.grid(color='grey', linestyle='--', linewidth=0.3)
plt.legend(ncol=3)
plt.savefig('fig14a.pdf')
        

plt.figure()
plt.bar(c_arr, nple, color='C0', width=0.5, alpha=0.25)
plt.xlabel(r'$Re$')
plt.ylabel(r'$n_p$')
plt.savefig('fig14b.pdf')



