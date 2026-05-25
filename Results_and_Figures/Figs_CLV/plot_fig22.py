# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
from scipy.spatial.distance import cosine

params = {'legend.fontsize': 9,
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

Angles = np.load("./data/L96_angles.npz")
principal_angles = Angles["principal_angles"]
_, nt = principal_angles.shape 
t0 = np.linspace(0,nt*0.01,nt)


plt.figure(figsize = (5,3))
plt.plot(t0,principal_angles[0],label=r"$\angle (\mathcal{M}^{u}, \mathcal{M}^{s})$")
plt.plot(t0,principal_angles[1],label=r"$\angle (\mathcal{M}^{u}, \mathcal{M}^{n})$")
plt.plot(t0,principal_angles[2],label=r"$\angle (\mathcal{M}^{n}, \mathcal{M}^{s})$")
plt.xlabel("time")
plt.ylabel('Angles (deg) between different manifolds')
plt.legend()
plt.grid()
plt.savefig("fig22a.pdf")

plt.figure(figsize = (3,3))
counts, bins = np.histogram(principal_angles[0], bins=100, density=True)
counts2, bins2 = np.histogram(principal_angles[1], bins=100, density=True)
counts3, bins3 = np.histogram(principal_angles[2], bins=100, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
bin_centers3 = (bins3[:-1] + bins3[1:]) / 2
pdf = counts #/ sum(counts)
pdf2 = counts2 #/ sum(counts2)
pdf3 = counts3 #/ sum(counts3)

plt.plot(bin_centers, pdf, label=r"$\min (\angle (\mathcal{M}^{u}, \mathcal{M}^{s}))$")
plt.plot(bin_centers2, pdf2, label=r"$\angle (\mathcal{M}^{u}, \mathcal{M}^{n})$")
plt.plot(bin_centers3, pdf3, label=r"$\angle (\mathcal{M}^{n}, \mathcal{M}^{s})$")
#plt.yscale("log")
plt.xlabel('Angles(deg)')
plt.title('Probability Density Function')
plt.grid()
plt.legend()
plt.savefig("fig22b.pdf")

