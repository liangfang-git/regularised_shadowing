
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


Lyapunov = np.load(f"./data/KolSol_CLVs_23.npz")
CLV = Lyapunov["CLV"]
ww = Lyapunov["wp"]
n_p = 5
CLVs = CLV[:,:,:n_p].copy()
CLVu = CLVs[:,:,:-2].copy()
CLVn = CLVs[:,:,n_p-2:n_p].copy()
nstep, nx, n_le = CLVs.shape
tCLVs = np.linspace(0,nstep*1,nstep)

print(nx)
principal_angles = np.zeros(nstep)
for i in range(nstep):
    C0 = CLVu[i].copy()
    C1 = CLVn[i].copy()
    Q0,_=np.linalg.qr(C0, 'reduced')
    Q1,R=np.linalg.qr(C1, 'reduced')
    _,S0,_ = np.linalg.svd(Q1.T @ Q0, full_matrices=False)
    principal_angles[i] = np.degrees(np.arccos(max(S0)))
 
ta = np.linspace(0,200,20)
principal_angles = principal_angles[1:].reshape(nstep//10,10)
fig, ax1 = plt.subplots(figsize=(4,3))
ax1.plot(ta, principal_angles[:,0],'o-',markersize=2.5,markerfacecolor='none',markeredgecolor='C0')
ax1.set_xlabel('t')
ax1.set_ylabel('Principal Angles', color='C0')
ax1.tick_params(colors='C0')
ax2 = ax1.twinx()
ax2.plot(ta, np.linalg.norm(ww,axis=1), 'o--',markersize=2.5,markerfacecolor='none',markeredgecolor='C3',color = "C3")
ax2.set_ylabel(r"$||\epsilon||$", color='C3')
ax2.tick_params(colors='C3')
# Display the legend
fig.tight_layout()
plt.savefig(f"fig25a.pdf",format='pdf')
df = pd.DataFrame({'Principal Angles' : principal_angles[:,0] , r'$||\epsilon||$':np.linalg.norm(ww,axis=1)})
plot = sns.jointplot(x='Principal Angles', y=r'$||\epsilon||$' ,height=3,data=df, fill = 1, kind = 'kde')
plot.plot_joint(sns.scatterplot, color="C1")
plt.savefig(f"fig25b.pdf",format='pdf')
    
  


