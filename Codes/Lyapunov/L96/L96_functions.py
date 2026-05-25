#-------------------------------------------------------------------------------------------#
# This source code contains functions defined related to the computation of the L96 system.  
# These functions are used for the (Augmented) NILSS algorithm for sensitivity calculation. 
# Author: Liang Fang
#         Postdoctoral Research Associate
#         Imperial College London
# Date:   2023-11-23
#-------------------------------------------------------------------------------------------#

# Lorenz 96 system 
# KS Equation:   du_i/dt = (u_{i+1}-u_{i-2})u_{i-1}-u_i+F
# with periodic bc 
# Space discretization: Finite Difference
# Time marching: RK45import numpy as np
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional

def l96_rhs(t, x, F):
    dxdt = np.zeros_like(x)
    N = len(x)
    for i in range(N):
        dxdt[i] = (x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F    
    return dxdt

def hom_linearised_l96_rhs(t, w, x):
    N = len(x)
    Ja = np.zeros((N, N))
    for i in range(N):
        Ja[i, i] = -1
        Ja[i, (i-1) % N] = x[(i+1) % N] - x[(i-2) % N]
        Ja[i, (i-2) % N] = -x[(i-1) % N]
        Ja[i, (i+1) % N] = x[(i-1) % N]
    dwdt = np.dot(Ja,w)
    return dwdt

def inhom_linearised_l96_rhs(t, v, x):
    N = len(x)
    Ja = np.zeros((N, N))
    for i in range(N):
        Ja[i, i] = -1
        Ja[i, (i-1) % N] = x[(i+1) % N] - x[(i-2) % N]
        Ja[i, (i-2) % N] = -x[(i-1) % N]
        Ja[i, (i+1) % N] = x[(i-1) % N]
    dvdt = np.dot(Ja,v)
    dfdrho = np.ones(N)
    dvdt += dfdrho
    return dvdt

def LE_ddt(t, x, F, nx, n_lyap):
    f = np.zeros_like(x)
    x_u = x[0:nx]
    x_w = x[nx:(n_lyap+1)*nx].reshape((nx,n_lyap))
    dudt = l96_rhs(t,x_u,F)
    dwdt = np.zeros_like(x_w)
    for i in range(n_lyap):
        dwdt[:,i] = hom_linearised_l96_rhs(t, x_w[:,i], x_u)
    f[:nx] = dudt.ravel()
    f[nx:] = dwdt.ravel()
    return f

class L96(object):

    def __init__(self, dt, L, nx, dx, precursor_time, args):
        self.precursor_time = precursor_time
        self.dt = dt
        self.L = L
        self.nx = nx
        self.dx = dx
        self.args = args
        self.x0 = np.random.rand(nx) 

    def precursor_simulation(self):
        F = self.args[0]
        t_span_develop = [0, self.precursor_time]
        nstep = int(round(self.precursor_time/self.dt))+1
        t_eval = np.linspace(0, self.precursor_time, nstep)
        sol0 = solve_ivp(lambda t, x: l96_rhs(t, x, F), t_span_develop, self.x0, method='RK45', t_eval=t_eval)
        xs = sol0.y.T[-1]
        return xs
    
    def nonlinear_simulation(self, T, xs: Optional[list[float]] = None):
        t_span = [0, T]
        nt = int(T/self.dt)+1
        t_eval = np.linspace(0, T, nt)
        F = self.args[0]
        sol = solve_ivp(lambda t, x: l96_rhs(t, x, F), t_span, xs, method='RK45', t_eval=t_eval)
        x_solution = sol.y.T
        t_sim = sol.t
        return x_solution, t_sim
    
    def LE_multiple_linear_simulation(self, t_start, t_end, w, n_lyap):
        F = self.args[0]
        t_span = [t_start, t_end]
        sol = solve_ivp(lambda t, x: LE_ddt(t, x, F, self.nx, n_lyap), t_span, w, method='RK45')
        w_solution = sol.y.T[-1]
        return w_solution
    
    def rhs_calculate(self, u):
        F = self.args[0]
        f = l96_rhs(None, u, F)
        return f
    

