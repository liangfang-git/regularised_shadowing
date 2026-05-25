# ------------------------------------------------------------------------------------------------------
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
import pickle

# Both Full trajectory design and Checkpoint design are considered in this code.
from Lyapunov_functions import * 
# Import the functions on different dynamical systems.
from KolSol_functions import *

# Parameters
dt = 0.01
nk = 16
precursor_time = 300
augmented = None
chi = 0.0
delta = 0

# total control time
T_all =200
# number of repeats
nr = 10
# List of the controls
c = 38.0
splitdT = 10
n_ple = 5
leading_le = 0.10627
max_tseg = 1.0/(leading_le)
dT = 10#round(max_tseg,2)

n_lyap = 5
stept = dT
t_forward_tran = 1000
t_forward_develop = 700
t_back_tran = 500 # The transient time for backward should be smaller than the forward develop time
ioutp = 50000

nseg = math.ceil(T_all/dT)
w_read = np.load('w_random.npz')
w_rand = w_read['w_rand']

os.makedirs(f"./results/", exist_ok=True)
os.makedirs(f"./new_initials/", exist_ok=True)
os.makedirs(f"./figures/", exist_ok=True)
print ("Done initial configuration, now move to calculations")

for r in range(5):
    print (f"For Reynold number {int(c)} with the case repeatnumber {r}")
    Simulation = KolSol(dt, nk, precursor_time, augmented, chi, delta, args=(c,None))
    file_name_initial = "./initials/initial_field_re_"+str(int(c))+"_sample_"+str(int(r))+"_grid_32.pkl"
    with open(file_name_initial, 'rb') as file:
        initial = pickle.load(file)
    u_temp = Simulation.fourier_to_phys(initial)
    u_ps = u_temp.flatten()
    LE_function = Lyapunov(n_lyap, round(stept/splitdT,1), t_forward_tran, t_forward_develop, t_back_tran, ioutp)       
    lambda_QR,Lyapunovxp,Texp,CLV,w_out = LE_function.Lyapunov_QR(Simulation,u_ps)   
    np.savez(f"./new_initials/Kol_38_{r}", u_initial = w_out[0])
    
    print (f"CLVs done!")
    np.savez(f"./results/NILSS_KolSol_CLVs_{r}",  CLV=CLV)
    
    del Simulation
    

