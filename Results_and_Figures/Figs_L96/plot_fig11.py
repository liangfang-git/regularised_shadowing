# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import norm
from scipy import stats
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
pylab.rcParams.update(params)


Sol1 = np.load("./data/NILSS_L96_5.npz")
Sol2 = np.load("./data/NILSS_L96_regularised_5.npz")

fd1 = np.loadtxt("./data/FD_derivative_J1_5.dat")
fd2 = np.loadtxt("./data/FD_derivative_J2_5.dat")

c_arr = Sol1['c_arr']
dJr1 = Sol1['dJr']
st10 = np.array(dJr1[0,:,:]).std(1,ddof=1)
st11 = np.array(dJr1[1,:,:]).std(1,ddof=1)
dJr2 = Sol2['dJr']
st20 = np.array(dJr2[0,:,:]).std(1,ddof=1)
st21 = np.array(dJr2[1,:,:]).std(1,ddof=1)

print(dJr1[0,0,:],dJr1[1,0,:])

color0 = "C0"
color1 = "C3"

plt.figure()
plt.plot(c_arr, dJr1[0,:,:], '+', markersize = 2, color = color0)
plt.plot(c_arr, dJr2[0,:,:], '+', markersize = 2,color = color1)
plt.plot(c_arr, np.mean(dJr1[0,:,:],axis=-1), '-+', markersize = 2, color = color0, lw = 1,label=r'Standard NILSS')
plt.plot(c_arr, np.mean(dJr2[0,:,:],axis=-1), '--+', markersize = 2, color = color1, lw = 1,label=r'Regularised NILSS')
plt.fill_between(c_arr, np.mean(dJr1[0,:,:],axis=-1)-st10, np.mean(dJr1[0,:,:],axis=-1)+st10, alpha=0.3, facecolor=color0)
plt.fill_between(c_arr, np.mean(dJr2[0,:,:],axis=-1)-st20, np.mean(dJr2[0,:,:],axis=-1)+st20, alpha=0.3, facecolor=color1)
plt.plot(c_arr, fd1, 'k-', label='Finite Difference')
plt.xlabel(r'$F$')
plt.ylabel(r'$dJ_1/dF$')
plt.grid(True)
plt.legend()
plt.savefig("fig11a.pdf",format='pdf')


plt.figure()
plt.plot(c_arr, dJr1[1,:,:], '+', markersize = 2,color = color0)
plt.plot(c_arr, dJr2[1,:,:], '+', markersize = 2,color = color1)
plt.plot(c_arr, np.mean(dJr1[1,:,:],axis=-1), '-+', markersize = 2, color = color0, lw = 1,label=r'Standard NILSS')
plt.plot(c_arr, np.mean(dJr2[1,:,:],axis=-1), '--+', markersize = 2, color = color1, lw = 1,label=r'Regularised NILSS')
plt.fill_between(c_arr, np.mean(dJr1[1,:,:],axis=-1)-st11, np.mean(dJr1[1,:,:],axis=-1)+st11, alpha=0.3, facecolor=color0)
plt.fill_between(c_arr, np.mean(dJr2[1,:,:],axis=-1)-st21, np.mean(dJr2[1,:,:],axis=-1)+st21, alpha=0.3, facecolor=color1)
plt.plot(c_arr, fd2, 'k-', label='Finite Difference')
plt.xlabel(r'$F$')
plt.ylabel(r'$dJ_2/dF$')
plt.grid(True)
plt.legend()
plt.savefig("fig11b.pdf",format='pdf')

