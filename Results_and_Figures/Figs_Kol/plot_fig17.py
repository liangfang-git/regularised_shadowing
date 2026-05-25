# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
params = {'legend.fontsize': 7,
          'figure.figsize': (4.5, 3),
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
#matplotlib.use('Agg')

index=0


Temp = np.load("./data/NILSS_KolSol_1_38.0.npz")
dJr_temp = Temp['dJ0']
nr = 100


c_lb = 34
c_ub = 38
c_dist = 1
c_nb = int((c_ub - c_lb) / c_dist + 1)
c_arr = np.round(np.linspace(c_lb, c_ub, c_nb), 1)

dJ0 = np.zeros((c_nb,nr))
dJ1 = np.zeros((c_nb,nr))
for i, c in enumerate(c_arr):
    Sol = np.load(f"./data/NILSS_KolSol_1_{c}.npz")
    dJ0[i,:] = Sol['dJ0'][0].copy()
    dJ1[i,:] = Sol['dJ'][0].copy()

st0 = np.array(dJ0).std(1,ddof=1)
st1 = np.array(dJ1).std(1,ddof=1)

m0 = np.mean(dJ0,axis=-1)
m1 = np.mean(dJ1,axis=-1)
color0 = "C0"
color1 = "C3"
plt.figure()
fdfd = 0.0161*np.ones(len(c_arr))
plt.plot(c_arr, m0, '+-', color = color0, lw = 1,label=r'Standard NILSS')
plt.plot(c_arr, m1, '+--', color = color1, lw = 1,label=r'Regularised NILSS')
plt.plot(c_arr, fdfd, '-', color = 'k')
plt.plot(c_arr, dJ0[:,:], '+', color = color0)
plt.plot(c_arr, dJ1[:,:], '+', color = color1)

plt.fill_between(c_arr, m0-st0, m0+st0, alpha= 0.3, facecolor=color0)
plt.fill_between(c_arr, m1-st1, m1+st1, alpha= 0.3, facecolor=color1)
plt.xlim([33.5,38.5])
plt.xlabel(r'$Re$')
plt.ylabel(r'$d\overline{J}/dRe$')
plt.grid(True)
plt.legend()
plt.savefig('fig17.pdf',format='pdf')


