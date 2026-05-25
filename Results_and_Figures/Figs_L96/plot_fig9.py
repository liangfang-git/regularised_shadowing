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
from matplotlib.ticker import LogLocator, FuncFormatter

    
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


LC = np.load("./data/L96_conditions.npz")
oc0 = LC["oc0"]
oc1 = LC["oc1"]
c0 = LC["c0"]
c1 = LC["c1"]
condition = LC["condition"]
pp = LC["pp"]

plt.figure(figsize=(4.35,4.35))
plt.plot(oc1,c1,'+')
plt.plot(oc0,c0,'+',color="C3")
plt.plot([10**5,3*10**7],[10**5,3*10**7],color='k',lw=0.75)
plt.xlabel("Condition numbers before regularisation")
plt.ylabel("Condition numbers after regularisation")
plt.xscale("log")
plt.yscale("log")
plt.xlim([10**4,3*10**9])
plt.ylim([10**2,3*10**7])
plt.savefig("fig9a.pdf")
plt.figure(figsize=(4.35,4.35))
_,x1,_ = plt.hist(np.log10(condition.flatten()), bins=50, density=True, alpha=0.65, color="C0")
_,x2,_ =plt.hist(np.log10(pp.flatten()), bins=50, density=True, alpha=0.65, color="C1")
plt.xlabel('Condition numbers')
def log_format(x, pos):
    return f'$10^{{{int(x)}}}$'
plt.gca().xaxis.set_major_formatter(FuncFormatter(log_format))
plt.grid(True)
plt.savefig("fig9b.pdf")


