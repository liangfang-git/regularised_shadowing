# ------------------------------------------------------------------------------------------------------
# This source code is to calculate the sensitivity of chaotic systems by NILSS algorithm.
# The theory of the NILSS method can be found in the JCP paper by Angxiu Ni @ Qiqi Wang. (10.1016/j.jcp.2017.06.033)  
# In this code, the sensitivity of two different cost functions are considered, including the mean energy and the mean enstrophy.
# Author: Liang Fang
#         Postdoctoral Research Associate
#         Imperial College London
# Date:   2023-11-23
# ------------------------------------------------------------------------------------------------------
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
import pickle

# Both Full trajectory design and Checkpoint design are considered in this code.
from NILSS import * 

# Import the functions on different dynamical systems.
from KolSol_functions import *

# Parameters
dt = 0.01
nk = 16
precursor_time = 300
augmented = None
chi = 0.0
delta = 10
# design method 'CheckPoint' and 'FullTrajectory'
design_method = 'FullTrajectory' 

# total control time
T_all = 100
# number of repeats
nr = 100
nus_temp = np.loadtxt("./data/positive_LE.txt")
l = np.loadtxt("./data/leading_LE.txt")
# List of the controls
c_lb = 34
c_ub = 38
c_dist = 1
c_nb = int((c_ub - c_lb) / c_dist + 1)
c_arr = np.round(np.linspace(c_lb, c_ub, c_nb), 1)

Jr = np.zeros((2,c_nb,nr))
dJr = np.zeros((2,c_nb,nr))
w_read = np.load('./data/w_random.npz')
w_rand = w_read['w_rand']
regularised = 1
plot_curve = False
T_seg = np.zeros(c_nb)
T_control = np.zeros(c_nb)
for i, c in enumerate(c_arr):
    if plot_curve:
        os.makedirs(f"./l_curve_plot/{int(c)}/", exist_ok=True)
    for r in range (0,nr):
        print(c)
        t0 = time()
        n_ple = int(nus_temp[i])
        max_tseg = 1.0/(l[i])
        dT = round(max_tseg,2)
        nseg = math.ceil(T_all/dT)
        T_seg[i] = dT
        T_control[i] = round(dT*nseg,2)
        print('----------Re = '+str(int(c))+'--case'+str(r)+'--regularised'+str(regularised)+'----------------')        
        Simulation = KolSol(dt, nk, precursor_time, augmented, chi, delta, args=(c,None))
        Sensitivity = NILSS(n_ple, dT, nseg, design_method)
        file_name_initial = "./data/initials/initial_field_re_"+str(int(c))+"_sample_"+str(int(r))+"_grid_32.pkl"
        with open(file_name_initial, 'rb') as file:
            initial = pickle.load(file)
        u_temp = Simulation.fourier_to_phys(initial)
        u_ps = u_temp.flatten()
        u,v,t_sim,ksi = Sensitivity.NILSS_v_calculation(Simulation,u_ps,w_rand[r],regularised,r,c,plot_curve) 
        Jr[0,i,r], dJr[0,i,r] = Sensitivity.NILSS_sensitivity(u,v,t_sim,ksi,Simulation.JJu_space_meanenergy) 
        Jr[1,i,r], dJr[1,i,r] = Sensitivity.NILSS_sensitivity(u,v,t_sim,ksi,Simulation.JJu_space_meanenstrophy)     
        del Simulation
        del Sensitivity
        t1 = time()-t0
        print("Wall clock time:"+str(t1))
        print (dJr[:,i,r])
    np.savez(f"NILSS_KolSol_{regularised}_{c}", Re = c_arr[i], J = Jr[:,i,:], dJ=dJr[:,i,:], T_seg= T_seg, T_control = T_control)
np.savez("NILSS_KolSol_regularised", c_arr = c_arr, Jr = Jr, dJr=dJr, T_seg= T_seg, T_control = T_control)

