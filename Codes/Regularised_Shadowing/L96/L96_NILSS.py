# ------------------------------------------------------------------------------------------------------
# This source code is to calculate the sensitivity of chaotic systems by NILSS algorithm.
# The theory of the NILSS method can be found in the JCP paper by Angxiu Ni @ Qiqi Wang. (10.1016/j.jcp.2017.06.033)  
# In this code, the sensitivity of two different cost functions are considered, including the mean state and the mean energy.
# J1 = 1/(TL) \int_0^T \int_0^L u dx dt;  J2 = 1/(TL) \int_0^T \int_0^L u^2 dx dt.
# Author: Liang Fang
#         Postdoctoral Research Associate
#         Imperial College London
# Date:   2023-11-23
# ------------------------------------------------------------------------------------------------------
import os
import math
import numpy as np

# Both Full trajectory design and Checkpoint design are considered in this code.
from NILSS import * 
# Import the functions on different dynamical systems.
from L96_functions import *


ly = np.load('./data/LE_information_L96_10_20.npz')
le = ly['leading_le']
nl = ly['nple']
# Parameters
dt = 0.01
nx = 40
L = 40
dx = L/(nx)
precursor_time = 200

# design method 'CheckPoint' and 'FullTrajectory'
design_method = 'FullTrajectory' 
# total control time
T_all = 40
# number of repeats
nr = 50

# List of the controls
c_lb = 10
c_ub = 20
c_dist = 1
c_nb = int((c_ub - c_lb) / c_dist + 1)
c_arr = np.linspace(c_lb, c_ub, c_nb)

Jr = np.zeros((2,c_nb,nr))
dJr = np.zeros((2,c_nb,nr))
u_initial = np.load('./data/L96_initials_10-20.npz')
u_initial = u_initial["u0"]
T_seg = np.zeros(c_nb)
T_control = np.zeros(c_nb)
w_read = np.load('./data/w_rand.npz')
w_rand = w_read['w_rand']
regularised = 1
plot_curve = False
for i, c in enumerate(c_arr):

    if plot_curve:
        os.makedirs(f"./l_curve_plot/{int(c)}/", exist_ok=True)
    n_ple = int(nl[i])
    max_tseg = 1.0/(le[i])
    dT = round(max_tseg,2)
    nseg = math.ceil(T_all/dT)
    T_seg[i] = dT
    T_control[i] = round(dT*nseg,2)
    for r in range (0,nr):
        print('----------F = '+str(int(c))+'--case'+str(r)+'--regularised'+str(regularised)+'----------------')        
        Simulation = L96(dt, L, nx, dx, precursor_time, args=(c,None))
        Sensitivity = NILSS(n_ple, dT, nseg, design_method)
        u,v,t_sim,ksi = Sensitivity.NILSS_v_calculation(Simulation,u_initial[r,i],w_rand,regularised,r,c,plot_curve)
        Jr[0,i,r], dJr[0,i,r] = Sensitivity.NILSS_sensitivity(u,v,t_sim,ksi,Simulation.JJu_space_meanu)
        Jr[1,i,r], dJr[1,i,r] = Sensitivity.NILSS_sensitivity(u,v,t_sim,ksi,Simulation.JJu_space_meanenergy)
        print (dJr[0,i,r],dJr[1,i,r])
        del Simulation
        del Sensitivity

np.savez(f"NILSS_L96_regularised", c_arr = c_arr, Jr = Jr, dJr=dJr, T_seg = T_seg, T_control = T_control)
