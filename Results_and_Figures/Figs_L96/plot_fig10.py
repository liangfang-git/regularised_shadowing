# plot regularised NILSS sensitivity versus the segment-length coefficient alpha, dT = alpha*round(1/lambda_1,2)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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


Sol = np.load("./data/NILSS_L96_regularised_dT.npz")
fd1 = np.loadtxt("./data/FD_derivative_J1_10.dat")
fd2 = np.loadtxt("./data/FD_derivative_J2_10.dat")

a_arr = Sol['dt_factors']
ic = int(Sol['c']) - 10    # FD data are on the forcing grid F = 10, 11, ..., 20
dJr = Sol['dJr']
st0 = np.array(dJr[0,:,:]).std(1,ddof=1)
st1 = np.array(dJr[1,:,:]).std(1,ddof=1)

color0 = "C0"
plt.figure()
plt.plot(a_arr, dJr[0,:,:], '+', markersize = 2, color = color0)
plt.plot(a_arr, np.mean(dJr[0,:,:],axis=-1), '-+', markersize = 2, color = color0, lw = 1,label=r'Regularised NILSS')
plt.fill_between(a_arr, np.mean(dJr[0,:,:],axis=-1)-st0, np.mean(dJr[0,:,:],axis=-1)+st0, alpha=0.3, facecolor=color0)
plt.plot(a_arr, fd1[ic]*np.ones_like(a_arr), 'k-', label='Finite Difference')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$dJ_1/dF$')
plt.ylim([-0.45,0.45])
plt.grid(True)
plt.legend()
plt.savefig("fig10a.pdf",format='pdf')


plt.figure()
plt.plot(a_arr, dJr[1,:,:], '+', markersize = 2,color = color0)
plt.plot(a_arr, np.mean(dJr[1,:,:],axis=-1), '-+', markersize = 2, color = color0, lw = 1,label=r'Regularised NILSS')
plt.fill_between(a_arr, np.mean(dJr[1,:,:],axis=-1)-st1, np.mean(dJr[1,:,:],axis=-1)+st1, alpha=0.3, facecolor=color0)
plt.plot(a_arr, fd2[ic]*np.ones_like(a_arr), 'k-', label='Finite Difference')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$dJ_2/dF$')
plt.grid(True)
plt.ylim([0,3.85])
plt.legend()
plt.savefig("fig10b.pdf",format='pdf')
