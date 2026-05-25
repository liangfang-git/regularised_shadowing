# -------------------------------------------------------------------------------------------------
# This source code is to calculate the LE and CLV of the chaotic system.
# Author: Liang Fang
#         Postdoctoral Research Associate
#         Imperial College London
# Date:   2023-11-23
# -------------------------------------------------------------------------------------------------
import numpy as np

# Import the functions on different dynamical systems.
from L96_functions import *
from Lyapunov_functions import * 

# Parameters
dt = 0.01
nx = 40
L = 40
dx = L/(nx)
precursor_time = 200
n_lyap = 20
# number of repeats
nr = 50

dT = 0.5
splitdT = 5

t_forward_tran = 1000
t_forward_develop = 240
t_back_tran = 200 # The transient time for backward should be smaller than the forward develop time
ioutp = 50000

# List of the controls
c_lb = 10
c_ub = 20
c_dist = 1
c_nb = int((c_ub - c_lb) / c_dist + 1)
c_arr = np.linspace(c_lb, c_ub, c_nb)

u_initial = np.load('./data/L96_initials_10-20.npz')
u_initial = u_initial["u0"]

for i, c in enumerate(c_arr):
    for r in range (0,nr):
        print('----------F = '+str(int(c))+'--case'+str(r)+'----------------')        
        Simulation = L96(dt, L, nx, dx, precursor_time, args=(c,None))
        LE_function = Lyapunov(n_lyap, round(dT/splitdT,1), t_forward_tran, t_forward_develop, t_back_tran, ioutp)       
        lambda_QR,Lyapunovxp,Texp,CLV,w_out = LE_function.Lyapunov_QR(Simulation,u_initial[r,i])   
        np.savez(f"./NILSS_L96_CLVs_{c}_{r}", CLV=CLV)
        del Simulation,LE_function

