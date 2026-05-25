# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
import os
from scipy.spatial.distance import cosine
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

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

#pylab.rcParams.update(params)


c = 38.0
r = 23
Lyapunov = np.load(f"./data/KolSol_CLVs_{r}.npz")

CLVs = Lyapunov["CLV"]
nstep, nx, n_le = CLVs.shape
CLVu = CLVs[:,:,:-2].copy()
CLVn = CLVs[:,:,n_le-2:n_le].copy()
print (nstep)


tCLVs = np.linspace(0,nstep*1,nstep)
principal_angles = np.zeros(nstep)
for i in range(nstep):
    C0 = CLVu[i].copy()
    C1 = CLVn[i].copy()
    Q0,_=np.linalg.qr(C0, 'reduced')
    Q1,R=np.linalg.qr(C1, 'reduced')
    _,S0,_ = np.linalg.svd(Q1.T @ Q0, full_matrices=False)
    principal_angles[i] = np.degrees(np.arccos(max(S0)))

plt.figure(figsize=(5,2.8))
plt.plot(tCLVs,principal_angles)
plt.xlabel("time")
plt.ylabel(r"$\angle \left(\mathcal{M}_{u},\mathcal{M}_{n}\right)$")
plt.grid()
plt.savefig("fig23a.pdf",format='pdf')
all_pangles =np.load("./data/Angles_Kol.npz")["all_pangles"]
plt.figure(figsize=(4,4))
plt.hist(np.array(all_pangles).flatten(), bins=100, density=True, alpha=0.65, color="C0")
plt.xlabel('Principal Angles')
plt.grid(True)
plt.savefig("fig23b.pdf",format='pdf')