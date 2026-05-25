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
plt.figure()
plt.plot(t, dJ0, '+', markersize=1, color = color0)
plt.plot(t, m0, '-o', markersize=2, markeredgecolor=color0, markerfacecolor='white',color = color0,label='Standard NILSS')
plt.fill_between(t, m0-st0, m0+st0, alpha= 0.18, facecolor=color0)
plt.plot(t, dJ1, '+', markersize=1, color = color1)
plt.plot(t, m1, '--o', markersize=2, markeredgecolor=color1, markerfacecolor='white',color = color1,label='Regularised NILSS')
plt.fill_between(t, m1-st1, m1+st1, alpha= 0.18, facecolor=color1)
plt.plot([t[0],t[-1]], [0.0161,0.0161], '--', color = 'k')
plt.ylim([-0.008,0.03])
plt.xlabel(r'$T$')
plt.ylabel(r'$\frac{d\overline{J}}{d Re}$')
plt.legend()
plt.grid(True)
plt.savefig('fig15.pdf',format = 'pdf')



