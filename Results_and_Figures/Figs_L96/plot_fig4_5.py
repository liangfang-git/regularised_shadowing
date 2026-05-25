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
import scipy.stats as stats
params = {'legend.fontsize': 9,
          'figure.figsize': (3.5, 2),
          'lines.linewidth': 1.2,
          'figure.dpi': 300,
         'axes.labelsize': 8,
         'axes.titlesize': 9,
         'xtick.labelsize':8,
         'ytick.labelsize':8,
         'figure.autolayout':True}
pylab.rcParams['mathtext.fontset'] = 'stix'
pylab.rcParams['font.family'] = 'DeJavu Serif'
pylab.rcParams['font.serif'] = ['Times New Roman']

c_lb = 10
c_ub = 20
c_dist = 1
c_nb = int((c_ub - c_lb) / c_dist + 1)
c_arr = np.round(np.linspace(c_lb, c_ub, c_nb), 1)

def handle_r( ):
    L = np.load(f"./data/l_curve.npz")
    kappa = L["kappa"]
    kmax = L["kmax"]
    sigma = L["sigma"]
    rho = L["rho"]    
    eta = L["eta"]    

    ikmax = np.argmax(kappa)
    if (kmax < 0):
        ikmax = 0
    index = np.where(kappa == kmax)[0][0]
    reg_param = np.zeros(100)
    S = 1*(sigma)**0.5
    reg_param[-1] = max(S[-1],S[0]*16*np.finfo(float).eps)
    ratio = (S[0]/reg_param[-1])**(1./(100-1))
    for i in range(1,100): 
        reg_param[-1*i-1] = ratio*reg_param[-i]
    reg_param = reg_param[::-1]

    if (kmax > 0):
        print(reg_param[index]**2)
        rp = 1/(reg_param[index]**2+sigma[-1])
        pp = (reg_param[index]**2+sigma[0])/(reg_param[index]**2+sigma[-1])
        reg_c = reg_param[index]**2
    else:
        print(reg_param[0]**2)
        reg_c = reg_param[0]**2
        pp = (reg_param[0]**2+sigma[0])/(reg_param[0]**2+sigma[-1])
        rp = 1/(reg_param[0]**2+sigma[-1])
    
    plt.figure(figsize=(6,3.5))

    plt.plot(np.arange(1,len(sigma)+1,1),sigma,'-o',markersize = 3,lw = 1., markerfacecolor='white', label="Original")
    plt.plot(np.arange(1,len(sigma)+1,1),reg_c+sigma,'--+',markersize = 3, lw=1.,label="Regularised")
    plt.plot([1,len(sigma)+1],[reg_c,reg_c],'-', lw=1.,color="k")
    plt.yscale("log")
    plt.xlabel(rf"$i$")
    plt.ylabel(rf"$\sigma_i$")
    plt.grid(color='grey', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fig5.pdf")    

    plt.figure(figsize=(3.5,3.5))
    plt.plot(reg_param,kappa,'-',markersize=2,lw=1)
    plt.plot([reg_param[0],reg_param[-1]],[0,0],'-',color = 'k')
    plt.plot(reg_param[ikmax],kappa[ikmax],'o',color = 'C3')
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\kappa$")
    plt.xscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"fig4b.pdf")
    
    plt.figure(figsize=(3.5,3.5))
    plt.plot(rho,eta,'-',markersize=2,lw=1)
    plt.plot(rho[ikmax],eta[ikmax],'o',color = 'C3')
    plt.xscale("log",base=10)
    plt.yscale("log",base=10)
    plt.xlabel(r"$\|\eta(\gamma)\|^2$")
    plt.ylabel(r"$\|\rho(\gamma)\|^2$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"fig4a.pdf")
    print (reg_param[ikmax])
handle_r()

