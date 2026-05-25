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

datas = np.load("./data/Angles_Kol.npz")
all_pangles = datas["all_pangles"]
angs_all = datas["angs_all"]


plt.figure(figsize=(4.4,4))
matrix = np.mean(angs_all,axis=0)
plt.imshow((matrix), cmap='Blues', interpolation='nearest', aspect='auto', vmin=0, vmax=90, extent=(0.5, 5.5, 5.5, 0.5))
plt.xticks([1,2,3,4,5])
plt.yticks([5,4,3,2,1])
plt.colorbar()
# Display row and column indices
# Add labels
x_start = 0.5
x_end = 5.5
y_start = 0.5
y_end = 5.5
size = 5        
jump_x = (x_end - x_start) / (2.0 * size)
jump_y = (y_end - y_start) / (2.0 * size)
x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        label = round(matrix[y_index, x_index],2)
        text_x = x + jump_x
        text_y = y + jump_y
        plt.text(text_x, text_y, label, color='black', ha='center', va='center')
plt.ylabel(r'CLV index ($j$)')
plt.xlabel(r'CLV index ($i$)')
plt.savefig(f"fig21a.pdf",format='pdf')

plt.figure(figsize=(4.4,4))
matrix = np.mean(np.partition(angs_all,201*5,axis=0)[:201*5],axis=0)
plt.imshow((matrix), cmap='Blues', interpolation='nearest', aspect='auto', vmin=0, vmax=90, extent=(0.5, 5.5, 5.5, 0.5))
# Add colorbar to show values
x_start = 0.5
x_end = 5.5
y_start = 0.5
y_end = 5.5
size = 5        
jump_x = (x_end - x_start) / (2.0 * size)
jump_y = (y_end - y_start) / (2.0 * size)
x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        label = round(matrix[y_index, x_index],2)
        text_x = x + jump_x
        text_y = y + jump_y
        plt.text(text_x, text_y, label, color='black', ha='center', va='center')
plt.xticks([1,2,3,4,5])
plt.yticks([5,4,3,2,1])
plt.colorbar()
# Display row and column indices
# Add labels
plt.ylabel(r'CLV index ($j$)')
plt.xlabel(r'CLV index ($i$)')
plt.savefig(f"fig21b.pdf",format='pdf')

plt.figure(figsize=(4.4,4))
matrix = np.mean(np.partition(angs_all,100*5,axis=0)[:100*5],axis=0)
plt.imshow((matrix), cmap='Blues', interpolation='nearest', aspect='auto', vmin=0, vmax=90, extent=(0.5, 5.5, 5.5, 0.5))
# Add colorbar to show values
x_start = 0.5
x_end = 5.5
y_start = 0.5
y_end = 5.5
size = 5        
jump_x = (x_end - x_start) / (2.0 * size)
jump_y = (y_end - y_start) / (2.0 * size)
x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        label = round(matrix[y_index, x_index],2)
        text_x = x + jump_x
        text_y = y + jump_y
        plt.text(text_x, text_y, label, color='black', ha='center', va='center')


plt.xticks([1,2,3,4,5])
plt.yticks([5,4,3,2,1])
plt.colorbar()
# Display row and column indices
# Add labels
plt.ylabel(r'CLV index ($j$)')
plt.xlabel(r'CLV index ($i$)')
plt.savefig(f"fig21c.pdf",format='pdf')

