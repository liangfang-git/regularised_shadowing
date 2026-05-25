#-------------------------------------------------------------------------------------------#
# This source code contains functions defined related to the computation of the kolmogorov flow.  
# These functions are used for the (Augmented) NILSS algorithm for sensitivity calculation. 
# Part of the code is from KolSol solver developed by MagriLab that can be found at:
# https://github.com/MagriLab/KolSol/tree/main
# The linearised NS part and some relative functions are developed by the author.
# Author: Liang Fang
#         Postdoctoral Research Associate
#         Imperial College London
# Date:   2023-11-23
#-------------------------------------------------------------------------------------------#


import itertools as it
import string
from typing import List, Optional
import einops
import numpy as np
import opt_einsum as oe
import enum

class eDirection(enum.IntEnum):
    i: int = 0
    j: int = 1
    k: int = 2

class KolSol(object):

    def __init__(self, dt: float, nk: int, precursor_time: float, augmented, chi, delta, args) -> None:

        """Kolmogorov Flow Solver Class.

        Implementation of a Fourier-Galerkin pseudospectral solver for the
        incompressible navier-stokes equations as described by Canuto, CH [7.2].

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        nf: int
            Prescribed forcing frequency.
        re: float
            Reynolds number of the flow.
        ndim: int, default=2
            Number of dimensions to solve for.
        """
        self.precursor_time = precursor_time
        self.dt = dt
        self.L = 2*np.pi
        self.nx = 2*(2*nk)**2
        self.dx = self.L/(2*nk)
        self.args = args
        self.augmented = augmented
        self.chi = chi
        self.delta = delta
        
        self.nk = nk
        self.nf = 4
        self.re = args[0]
        self.ndim = 2

        self.nk_grid = 2 * self.nk
        self.mk_grid = 4 * self.nk

        x = np.linspace(0.0, 2.0 * np.pi, self.nk_grid + 1)[:-1]
        self.xt = np.stack(np.meshgrid(*(x for _ in range(self.ndim)), indexing='ij'), axis=-1)

        k = np.fft.fftshift(np.fft.fftfreq(self.nk_grid, 1 / self.nk_grid))
        self.kt = np.stack(np.meshgrid(*(k for _ in range(self.ndim)), indexing='ij'), axis=-1)
        self.kk = np.sum(np.power(self.kt, 2), axis=-1)

        self.nabla = 1j * self.kt
        self.f = np.zeros((*(self.nk_grid for _ in range(self.ndim)), self.ndim), dtype=np.complex128)
        self.f[..., eDirection.i] = np.fft.fftshift(np.fft.fftn(np.sin(self.nf * self.xt[..., eDirection.j])))

    def generate_dwdu(self):
        # Avoid division by zero by checking if kt[:,:,0] and kt[:,:,1] are not zero
        nonzero_indices0 = (abs(self.kt[:,:,0]) > 1e-10) 
        nonzero_indices1 = (abs(self.kt[:,:,1]) > 1e-10) 
        kkt = np.zeros_like(self.kt,dtype=complex)
        # Update kkt[:,:,0] and kkt[:,:,1] based on the provided formulae
        with np.errstate(divide='ignore', invalid='ignore'):
            kkt[:,:,0] = np.where(nonzero_indices0, self.kt[:,:,0] + (self.kt[:,:,1]**2) / self.kt[:,:,0], 0)
            kkt[:,:,1] = np.where(nonzero_indices1, self.kt[:,:,1] + (self.kt[:,:,0]**2) / self.kt[:,:,1], 0)
        return kkt
    
    def precursor_simulation(self):
        field_hat = self.random_field(magnitude=1.0, sigma=2.0)
        u0 = self.fourier_to_phys(field_hat)
        uv = u0.flatten()
        xs = self.dynamics_forward_R45_onlyfinal(uv, int(round(self.precursor_time / self.dt)) + 1, self.dt)
        return xs
    
    def nonlinear_simulation(self, T, xs: Optional[list[float]] = None):
        #if not xs:
        #    xs = self.precursor_simulation()
        nt = int(T/self.dt)+1
        t_eval = np.linspace(0, T, nt)
        x_solution = self.dynamics_forward_R45(xs, int(round(T / self.dt)) + 1, self.dt)
        return x_solution, t_eval
    
    def LE_multiple_linear_simulation(self, t_start, t_end, w, n_lyap):
        if (self.augmented):
            w_solution = self.dynamics_forward_R45_LE_augmented(w, int(round((t_end-t_start) / self.dt)) + 1, n_lyap, self.dt)
        else:
            w_solution = self.dynamics_forward_R45_LE(w, int(round((t_end-t_start) / self.dt)) + 1, n_lyap, self.dt)
        return w_solution
    
    def NILSS_forward(self, t_start, t_end, x0, nstep, n_ple):
        t_eval = np.linspace(t_start, t_end, nstep)
        if (self.augmented):
            x1 =  self.dynamics_all_forward_R45_augmented(x0, int(round((t_end-t_start) / self.dt)) + 1, n_ple, self.dt)
        else:
            x1 =  self.dynamics_all_forward_R45(x0, int(round((t_end-t_start) / self.dt)) + 1, n_ple, self.dt)
        return x1, t_eval

    def rhs_calculate(self, u: np.ndarray) -> np.ndarray:
        u = np.reshape(u,(2*self.nk,2*self.nk,2))
        u_hat = self.phys_to_fourier(u)
        du_hat_dt = self.dynamics(u_hat)
        dudt = self.fourier_to_phys(du_hat_dt)
        return dudt.flatten()   
    
    def JJu_space_meanenergy(self, u: np.ndarray, v: np.ndarray):
        u = np.reshape(u,(2*self.nk,2*self.nk,2))
        u1 = u[...,0]
        u2 = u[...,1]
        v = np.reshape(v,(2*self.nk,2*self.nk,2))
        v1 = v[...,0]
        v2 = v[...,1]
        Jt = 0.5*np.mean(u1**2+u2**2, axis=(0, 1,))
        dJt = np.mean(u1*v1+u2*v2, axis=(0, 1,))
        return Jt, dJt

    def JJu_space_meanenstrophy(self, u: np.ndarray, v: np.ndarray):
        u = np.reshape(u,(2*self.nk,2*self.nk,2))
        u_hat = self.phys_to_fourier(u)
        vor_hat = np.zeros_like(u_hat,dtype=complex)
        vor_hat[...,0] = np.cross(self.nabla, u_hat)
        vor = self.fourier_to_phys(vor_hat)
        v = np.reshape(v,(2*self.nk,2*self.nk,2))
        v_hat = self.phys_to_fourier(v)
        dvor_hat = np.zeros_like(u_hat,dtype=complex)
        dvor_hat[...,0] = np.cross(self.nabla, v_hat)
        dvor = self.fourier_to_phys(dvor_hat)
        Jt = np.mean(vor[...,0]**2, axis=(0, 1,))
        dJt = 2*np.mean(vor[...,0]*dvor[...,0], axis=(0, 1,))
        return Jt, dJt
    
    def objective_all(self, u: np.ndarray) -> np.float128:
        u = np.reshape(u,(2*self.nk,2*self.nk,2))
        u1 = u[...,0]
        u2 = u[...,1]
        Jt = 0.5*np.mean(u1**2+u2**2, axis=(0, 1,))
        u_hat = self.phys_to_fourier(u)
        vor_hat = np.zeros_like(u_hat,dtype=complex)
        vor_hat[...,0] = np.cross(self.nabla, u_hat)
        vor = self.fourier_to_phys(vor_hat)
        Jt2 = np.mean(vor[...,0]**2, axis=(0, 1,))
        return Jt,Jt2,vor[...,0]
        
    def objective(self, u: np.ndarray) -> np.float128:
        u = np.reshape(u,(2*self.nk,2*self.nk,2))
        u1 = u[...,0]
        u2 = u[...,1]
        J = 0.5*(np.mean(u1**2, axis=(0, 1,))+np.mean(u2**2, axis=(0, 1,)))
        return J

    def dynamics(self, u_hat: np.ndarray) -> np.ndarray:

        aapt = np.array([
            [self.aap(u_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])

        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        with np.errstate(divide='ignore', invalid='ignore'):
            k_dot_f = oe.contract('...u, ...u -> ...', self.kt, f_hat + self.f) / self.kk
            k_dot_f[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk = oe.contract('...u, ... -> ...u', self.kt, k_dot_f)

        # Canuto EQ [7.2.11]
        du_hat_dt = (f_hat + self.f) - kk_fk - (1.0 / self.re) * oe.contract('...u, ... -> ...u', u_hat, self.kk)

        return du_hat_dt
    
    def dynamics_forward_R45(self, uv: np.ndarray, nt: np.int64, dt: np.float128) -> np.ndarray:
        ur = np.zeros((nt,self.nk_grid*self.nk_grid*2))
        u = np.reshape(uv,(self.nk_grid,self.nk_grid,2))
        ur[0,:] = uv
        u_hat = self.phys_to_fourier(u)
        u_hat_array = np.zeros(shape=(4, self.nk_grid, self.nk_grid, self.ndim),dtype=np.complex128)
        for t in range(nt-1):
            u_hat_array[0, ...] = u_hat
            k1 = self.dynamics(u_hat_array[0, ...])
            u_hat_array[1, ...] = u_hat + 0.5 * k1 * dt
            k2 = self.dynamics(u_hat_array[1, ...])
            u_hat_array[2, ...] = u_hat + 0.5 * k2 * dt
            k3 = self.dynamics(u_hat_array[2, ...])
            u_hat_array[3, ...] = u_hat + 1.0 * k3 * dt
            k4 = self.dynamics(u_hat_array[3, ...])
            u_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            ur_temp = self.fourier_to_phys(u_hat)
            ur[t+1] = ur_temp.flatten() 
        return ur
    
    def dynamics_forward_R45_onlyfinal(self, uv: np.ndarray, nt: np.int64, dt: np.float128) -> np.ndarray:
        u = np.reshape(uv,(self.nk_grid,self.nk_grid,2))
        u_hat = self.phys_to_fourier(u)
        u_hat_array = np.zeros(shape=(4, self.nk_grid, self.nk_grid, self.ndim),dtype=np.complex128)
        for t in range(nt):
            u_hat_array[0, ...] = u_hat
            k1 = self.dynamics(u_hat_array[0, ...])
            u_hat_array[1, ...] = u_hat + 0.5 * k1 * dt
            k2 = self.dynamics(u_hat_array[1, ...])
            u_hat_array[2, ...] = u_hat + 0.5 * k2 * dt
            k3 = self.dynamics(u_hat_array[2, ...])
            u_hat_array[3, ...] = u_hat + 1.0 * k3 * dt
            k4 = self.dynamics(u_hat_array[3, ...])
            u_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        ur = self.fourier_to_phys(u_hat)
        return ur.flatten()
    
    def linearised_dynamics(self, u_hat: np.ndarray, pu_hat: np.ndarray) -> np.ndarray:
        # Canuto EQ [7.2.12]
        aaptpu = np.array([
            [self.aap(pu_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])

        nonlinear = oe.contract('...t, ut... -> ...u', -self.nabla, aaptpu) + oe.contract('...t, tu... -> ...u', -self.nabla, aaptpu)        
        with np.errstate(divide='ignore', invalid='ignore'):
            k_dot_f_p = oe.contract('...u, ...u -> ...', self.kt, nonlinear) / self.kk
            k_dot_f_p[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk_p = oe.contract('...u, ... -> ...u', self.kt, k_dot_f_p)

        # Canuto EQ [7.2.11]
        dpu_hat_dt = nonlinear - kk_fk_p - (1.0 / self.re) * oe.contract('...u, ... -> ...u', pu_hat, self.kk)

        return dpu_hat_dt
    
    def aug_linearised_dynamics(self, u_hat: np.ndarray, pu_hat_all: np.ndarray, chi: float, deltat: float) -> np.ndarray:
        # Canuto EQ [7.2.12]
        pu_hat = pu_hat_all[...,0]
        dpu_hat_dt_aug = np.zeros_like(pu_hat_all,dtype=np.complex128)
        aaptpu = np.array([
            [self.aap(pu_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])
        nonlinear = oe.contract('...t, ut... -> ...u', -self.nabla, aaptpu) + oe.contract('...t, tu... -> ...u', -self.nabla, aaptpu)        
        with np.errstate(divide='ignore', invalid='ignore'):
            k_dot_f_p = oe.contract('...u, ...u -> ...', self.kt, nonlinear) / self.kk
            k_dot_f_p[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk_p = oe.contract('...u, ... -> ...u', self.kt, k_dot_f_p)

        # Canuto EQ [7.2.11]
        dpu_hat_dt = nonlinear - kk_fk_p - (1.0 / self.re) * oe.contract('...u, ... -> ...u', pu_hat, self.kk)
        dpu_hat_dt_aug[..., 0] = dpu_hat_dt - chi*(pu_hat_all[..., 0] - pu_hat_all[..., 1])
        dpu_hat_dt_aug[..., 1] = 1.0/deltat*(pu_hat_all[..., 0] - pu_hat_all[..., 1])
        return dpu_hat_dt_aug    
       
    def inhom_linearised_dynamics(self, u_hat: np.ndarray, pu_hat: np.ndarray) -> np.ndarray:
        aaptpu = np.array([
            [self.aap(pu_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])

        nonlinear = oe.contract('...t, ut... -> ...u', -self.nabla, aaptpu) + oe.contract('...t, tu... -> ...u', -self.nabla, aaptpu)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            k_dot_f_p = oe.contract('...u, ...u -> ...', self.kt, nonlinear) / self.kk
            k_dot_f_p[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk_p = oe.contract('...u, ... -> ...u', self.kt, k_dot_f_p)

        # Canuto EQ [7.2.11]
        dpu_hat_dt = nonlinear - kk_fk_p - (1.0 / self.re) * oe.contract('...u, ... -> ...u', pu_hat, self.kk) + (1.0 / self.re)**2 * oe.contract('...u, ... -> ...u', u_hat, self.kk)

        return dpu_hat_dt
    
    def aug_inhom_linearised_dynamics(self, u_hat: np.ndarray, pu_hat_all: np.ndarray, chi: float, deltat: float) -> np.ndarray:
        pu_hat = pu_hat_all[...,0]
        dpu_hat_dt_aug = np.zeros_like(pu_hat_all,dtype=np.complex128)
        aaptpu = np.array([
            [self.aap(pu_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])

        nonlinear = oe.contract('...t, ut... -> ...u', -self.nabla, aaptpu) + oe.contract('...t, tu... -> ...u', -self.nabla, aaptpu)        
        with np.errstate(divide='ignore', invalid='ignore'):
            k_dot_f_p = oe.contract('...u, ...u -> ...', self.kt, nonlinear) / self.kk
            k_dot_f_p[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk_p = oe.contract('...u, ... -> ...u', self.kt, k_dot_f_p)
        dpu_hat_dt = nonlinear - kk_fk_p - (1.0 / self.re) * oe.contract('...u, ... -> ...u', pu_hat, self.kk) + (1.0 / self.re)**2 * oe.contract('...u, ... -> ...u', u_hat, self.kk)
        dpu_hat_dt_aug[..., 0] = dpu_hat_dt - chi*(pu_hat_all[..., 0] - pu_hat_all[..., 1])
        dpu_hat_dt_aug[..., 1] = 1.0/deltat*(pu_hat_all[..., 0] - pu_hat_all[..., 1])
        return dpu_hat_dt_aug
    
    def dynamics_forward_R45_LE(self, uv_all: np.ndarray, nt: np.int64, n_lyap: np.int64, dt: np.float128) -> np.ndarray:
        u = uv_all[0:self.nx]
        w_temp = uv_all[self.nx:(n_lyap+1)*self.nx]
        w = np.reshape(w_temp,(self.nx,n_lyap))
        sol = np.zeros(n_lyap*self.nx+self.nx)
        sol[:] = uv_all[:]

        u = np.reshape(u,(self.nk_grid,self.nk_grid,2))
        u_hat = self.phys_to_fourier(u)
        u_hat_array = np.zeros(shape=(5, self.nk_grid, self.nk_grid, self.ndim),dtype=np.complex128)
        w_hat_all = np.zeros(shape=(self.nk_grid, self.nk_grid, self.ndim, n_lyap),dtype=np.complex128)
        for i in range(n_lyap):
            w_tempf = np.reshape(w[:,i],(self.nk_grid,self.nk_grid,2))
            w_hat_all[...,i] = self.phys_to_fourier(w_tempf)
            w_hat_all[...,i] = self.divergence_free(w_hat_all[...,i])

        for t in range(nt-1):
            wv_all = np.zeros((self.nx,n_lyap))
            u_hat_array[1, ...] = u_hat
            k1 = self.dynamics(u_hat_array[1, ...])
            u_hat_array[2, ...] = u_hat + 0.5 * k1 * dt
            k2 = self.dynamics(u_hat_array[2, ...])
            u_hat_array[3, ...] = u_hat + 0.5 * k2 * dt
            k3 = self.dynamics(u_hat_array[3, ...])
            u_hat_array[4, ...] = u_hat + 1.0 * k3 * dt
            k4 = self.dynamics(u_hat_array[4, ...])
            u_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            ur = self.fourier_to_phys(u_hat)
            sol[:self.nx] = ur.flatten()
            for i in range(n_lyap):
                p1 = self.linearised_dynamics(u_hat_array[1, ...],w_hat_all[...,i])
                p2 = self.linearised_dynamics(u_hat_array[2, ...],w_hat_all[...,i] + 0.5 * p1 * dt)
                p3 = self.linearised_dynamics(u_hat_array[3, ...],w_hat_all[...,i] + 0.5 * p2 * dt)
                p4 = self.linearised_dynamics(u_hat_array[4, ...],w_hat_all[...,i] + 1.0 * p3 * dt)
                w_hat_all[...,i] += dt * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0
                w2 = self.fourier_to_phys(w_hat_all[...,i])
                wv_all[:,i] = w2.flatten()
            sol[self.nx:(n_lyap+1)*self.nx] = wv_all.flatten()
        return sol

    def dynamics_forward_R45_LE_augmented(self, uv_all: np.ndarray, nt: np.int64, n_lyap: np.int64, dt: np.float128) -> np.ndarray:
        u = uv_all[0:self.nx]
        w_temp = uv_all[self.nx:(2*n_lyap+1)*self.nx]
        w = np.reshape(w_temp,(2*self.nx,n_lyap))
        sol = np.zeros(n_lyap*2*self.nx+self.nx)
        sol[:] = uv_all[:]

        u = np.reshape(u,(self.nk_grid,self.nk_grid,2))
        u_hat = self.phys_to_fourier(u)
        u_hat_array = np.zeros(shape=(5, self.nk_grid, self.nk_grid, self.ndim),dtype=np.complex128)
        w_hat_all = np.zeros(shape=(self.nk_grid, self.nk_grid, self.ndim, 2, n_lyap),dtype=np.complex128)
        for i in range(n_lyap):
            w_tempf = np.reshape(w[:self.nx,i],(self.nk_grid,self.nk_grid,2))
            w_hat_all[...,0,i] = self.phys_to_fourier(w_tempf)
            w_hat_all[...,0,i] = self.divergence_free(w_hat_all[...,0,i])
            w_tempf2 = np.reshape(w[self.nx:,i],(self.nk_grid,self.nk_grid,2))
            w_hat_all[...,1,i] = self.phys_to_fourier(w_tempf2)
            w_hat_all[...,1,i] = self.divergence_free(w_hat_all[...,1,i])
        for t in range(nt-1):
            wv_all = np.zeros((self.nx,n_lyap))
            u_hat_array[1, ...] = u_hat
            k1 = self.dynamics(u_hat_array[1, ...])
            u_hat_array[2, ...] = u_hat + 0.5 * k1 * dt
            k2 = self.dynamics(u_hat_array[2, ...])
            u_hat_array[3, ...] = u_hat + 0.5 * k2 * dt
            k3 = self.dynamics(u_hat_array[3, ...])
            u_hat_array[4, ...] = u_hat + 1.0 * k3 * dt
            k4 = self.dynamics(u_hat_array[4, ...])
            u_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            ur = self.fourier_to_phys(u_hat)
            sol[:self.nx] = ur.flatten()
            for i in range(n_lyap):
                p1 = self.aug_linearised_dynamics(u_hat_array[1, ...],w_hat_all[...,i],self.chi,self.delta)
                p2 = self.aug_linearised_dynamics(u_hat_array[2, ...],w_hat_all[...,i] + 0.5 * p1 * dt,self.chi,self.delta)
                p3 = self.aug_linearised_dynamics(u_hat_array[3, ...],w_hat_all[...,i] + 0.5 * p2 * dt,self.chi,self.delta)
                p4 = self.aug_linearised_dynamics(u_hat_array[4, ...],w_hat_all[...,i] + 1.0 * p3 * dt,self.chi,self.delta)
                w_hat_all[...,i] += dt * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0
                w2 = self.fourier_to_phys(w_hat_all[...,0,i])
                wv_all[:self.nx,i] = w2.flatten()
                w2 = self.fourier_to_phys(w_hat_all[...,1,i])
                wv_all[self.nx:,i] = w2.flatten()
            sol[self.nx:(2*n_lyap+1)*self.nx] = wv_all.flatten()
        return sol

    def dynamics_all_forward_R45(self, uv_all: np.ndarray, nt: np.int64, n_ple: np.int64, dt: np.float128) -> np.ndarray:
        u = uv_all[0:self.nx]
        w_temp = uv_all[self.nx:(n_ple+1)*self.nx]
        w = np.reshape(w_temp,(self.nx,n_ple))
        v = uv_all[(n_ple+1)*self.nx:]
        sol = np.zeros((nt,n_ple*self.nx+2*self.nx))
        sol[0,:] = uv_all[:]

        u = np.reshape(u,(self.nk_grid,self.nk_grid,2))
        u_hat = self.phys_to_fourier(u)
        u_hat_array = np.zeros(shape=(5, self.nk_grid, self.nk_grid, self.ndim),dtype=np.complex128)
        w_hat_all = np.zeros(shape=(self.nk_grid, self.nk_grid, self.ndim, n_ple),dtype=np.complex128)
        for i in range(n_ple):
            w_tempf = np.reshape(w[:,i],(self.nk_grid,self.nk_grid,2))
            w_hat_all[...,i] = self.phys_to_fourier(w_tempf)
            w_hat_all[...,i] = self.divergence_free(w_hat_all[...,i])
        v = np.reshape(v,(self.nk_grid,self.nk_grid,2))
        v_hat = self.phys_to_fourier(v)
        v_hat = self.divergence_free(v_hat)

        for t in range(nt-1):
            wv_all = np.zeros((self.nx,n_ple))
            u_hat_array[1, ...] = u_hat
            k1 = self.dynamics(u_hat_array[1, ...])
            u_hat_array[2, ...] = u_hat + 0.5 * k1 * dt
            k2 = self.dynamics(u_hat_array[2, ...])
            u_hat_array[3, ...] = u_hat + 0.5 * k2 * dt
            k3 = self.dynamics(u_hat_array[3, ...])
            u_hat_array[4, ...] = u_hat + 1.0 * k3 * dt
            k4 = self.dynamics(u_hat_array[4, ...])
            u_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            ur = self.fourier_to_phys(u_hat)
            sol[t+1,:self.nx] = ur.flatten()
            for i in range(n_ple):
                p1 = self.linearised_dynamics(u_hat_array[1, ...],w_hat_all[...,i])
                p2 = self.linearised_dynamics(u_hat_array[2, ...],w_hat_all[...,i] + 0.5 * p1 * dt)
                p3 = self.linearised_dynamics(u_hat_array[3, ...],w_hat_all[...,i] + 0.5 * p2 * dt)
                p4 = self.linearised_dynamics(u_hat_array[4, ...],w_hat_all[...,i] + 1.0 * p3 * dt)
                w_hat_all[...,i] += dt * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0
                w2 = self.fourier_to_phys(w_hat_all[...,i])
                wv_all[:,i] = w2.flatten()
            sol[t+1,self.nx:(n_ple+1)*self.nx] = wv_all.flatten()
            p1 = self.inhom_linearised_dynamics(u_hat_array[1, ...],v_hat)
            p2 = self.inhom_linearised_dynamics(u_hat_array[2, ...],v_hat + 0.5 * p1 * dt)
            p3 = self.inhom_linearised_dynamics(u_hat_array[3, ...],v_hat + 0.5 * p2 * dt)
            p4 = self.inhom_linearised_dynamics(u_hat_array[4, ...],v_hat + 1.0 * p3 * dt)
            v_hat += dt * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0
            v = self.fourier_to_phys(v_hat)
            sol[t+1,(n_ple+1)*self.nx:] = v.flatten()
        return sol

    def dynamics_all_forward_R45_augmented(self, uv_all: np.ndarray, nt: np.int64, n_ple: np.int64, dt: np.float128) -> np.ndarray:
        u = uv_all[0:self.nx]
        w_temp = uv_all[self.nx:(2*n_ple+1)*self.nx]
        w = np.reshape(w_temp,(2*self.nx,n_ple))
        v = uv_all[(2*n_ple+1)*self.nx:]
        sol = np.zeros((nt,n_ple*2*self.nx+3*self.nx))
        sol[0,:] = uv_all[:]


        u = np.reshape(u,(self.nk_grid,self.nk_grid,2))
        u_hat = self.phys_to_fourier(u)
        u_hat_array = np.zeros(shape=(5, self.nk_grid, self.nk_grid, self.ndim),dtype=np.complex128)
        w_hat_all = np.zeros(shape=(self.nk_grid, self.nk_grid, self.ndim, 2, n_ple),dtype=np.complex128)
        v_hat = np.zeros(shape=(self.nk_grid, self.nk_grid, self.ndim, 2),dtype=np.complex128)
        for i in range(n_ple):
            w_tempf = np.reshape(w[:self.nx,i],(self.nk_grid,self.nk_grid,2))
            w_hat_all[...,0,i] = self.phys_to_fourier(w_tempf)
            w_hat_all[...,0,i] = self.divergence_free(w_hat_all[...,0,i])
            w_tempf2 = np.reshape(w[self.nx:,i],(self.nk_grid,self.nk_grid,2))
            w_hat_all[...,1,i] = self.phys_to_fourier(w_tempf2)
            w_hat_all[...,1,i] = self.divergence_free(w_hat_all[...,1,i])
        v_temp = np.reshape(v[:self.nx],(self.nk_grid,self.nk_grid,2))
        v_hat[...,0] = self.phys_to_fourier(v_temp)
        v_hat[...,0] = self.divergence_free(v_hat[...,0])
        v_temp = np.reshape(v[self.nx:],(self.nk_grid,self.nk_grid,2))
        v_hat[...,1] = self.phys_to_fourier(v_temp)
        v_hat[...,1] = self.divergence_free(v_hat[...,1])
        for t in range(nt-1):
            wv_all = np.zeros((self.nx,n_ple))
            u_hat_array[1, ...] = u_hat
            k1 = self.dynamics(u_hat_array[1, ...])
            u_hat_array[2, ...] = u_hat + 0.5 * k1 * dt
            k2 = self.dynamics(u_hat_array[2, ...])
            u_hat_array[3, ...] = u_hat + 0.5 * k2 * dt
            k3 = self.dynamics(u_hat_array[3, ...])
            u_hat_array[4, ...] = u_hat + 1.0 * k3 * dt
            k4 = self.dynamics(u_hat_array[4, ...])
            u_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            ur = self.fourier_to_phys(u_hat)
            sol[t+1,:self.nx] = ur.flatten()
            for i in range(n_ple):
                p1 = self.aug_linearised_dynamics(u_hat_array[1, ...],w_hat_all[...,i],self.chi,self.delta)
                p2 = self.aug_linearised_dynamics(u_hat_array[2, ...],w_hat_all[...,i] + 0.5 * p1 * dt,self.chi,self.delta)
                p3 = self.aug_linearised_dynamics(u_hat_array[3, ...],w_hat_all[...,i] + 0.5 * p2 * dt,self.chi,self.delta)
                p4 = self.aug_linearised_dynamics(u_hat_array[4, ...],w_hat_all[...,i] + 1.0 * p3 * dt,self.chi,self.delta)
                w_hat_all[...,i] += dt * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0
                w2 = self.fourier_to_phys(w_hat_all[...,0,i])
                wv_all[:self.nx,i] = w2.flatten()
                w2 = self.fourier_to_phys(w_hat_all[...,1,i])
                wv_all[self.nx:,i] = w2.flatten()
            sol[t+1,self.nx:(2*n_ple+1)*self.nx] = wv_all.flatten()
            p1 = self.aug_inhom_linearised_dynamics(u_hat_array[1, ...],v_hat,self.chi,self.delta)
            p2 = self.aug_inhom_linearised_dynamics(u_hat_array[2, ...],v_hat + 0.5 * p1 * dt,self.chi,self.delta)
            p3 = self.aug_inhom_linearised_dynamics(u_hat_array[3, ...],v_hat + 0.5 * p2 * dt,self.chi,self.delta)
            p4 = self.aug_inhom_linearised_dynamics(u_hat_array[4, ...],v_hat + 1.0 * p3 * dt,self.chi,self.delta)
            v_hat += dt * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0
            v_temp1 = self.fourier_to_phys(v_hat[...,0])
            v_temp2 = self.fourier_to_phys(v_hat[...,1])
            sol[t+1,(2*n_ple+1)*self.nx:(2*n_ple+2)*self.nx] = v_temp1.flatten()
            sol[t+1,(2*n_ple+2)*self.nx:(2*n_ple+3)*self.nx] = v_temp2.flatten()
        return sol


    def aap(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        n_leading_dims = f1.ndim - self.ndim
        leading_dims = f1.shape[:n_leading_dims]

        lb, ub = self.nk, 3 * self.nk
        scaling = (self.mk_grid / self.nk_grid) ** self.ndim

        fhat = np.stack((f1, f2), axis=-1)

        fhat_padded = np.zeros(([*leading_dims] + [self.mk_grid for _ in range(self.ndim)] + [2]), dtype=np.complex128)
        fhat_padded[tuple([...]) + tuple(slice(lb, ub) for _ in range(self.ndim)) + tuple([slice(None)])] = fhat

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, self.ndim + n_leading_dims

        f_phys = np.fft.irfftn(
            np.fft.ifftshift(fhat_padded, axes=range(axs_lb, axs_ub)),
            s=fhat_padded.shape[slice(axs_lb, axs_ub)],
            axes=range(axs_lb, axs_ub)
        )

        f1f2_hat_padded = np.fft.fftn(
            np.prod(f_phys, axis=-1),
            s=f_phys.shape[slice(axs_lb, axs_ub)],
            axes=range(axs_lb, axs_ub)
        )

        f1f2_hat_padded = scaling * np.fft.fftshift(f1f2_hat_padded, axes=range(axs_lb, axs_ub))
        f1f2_hat = f1f2_hat_padded[tuple([...]) + tuple(slice(lb, ub) for _ in range(self.ndim))]

        return f1f2_hat

    def dissip(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate dissipation of the velocity field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        np.ndarray
            Dissipation of the velocity field.
        """

        w_hat = self.vorticity(u_hat)

        # generate indices for each dimension
        g = iter(string.ascii_letters)
        idx = ''.join(next(g) for _ in range(self.ndim))

        dissipation = oe.contract(f'...{idx} -> ...',  w_hat * np.conj(w_hat))
        dissipation /= (self.re * self.nk_grid ** (2 * self.ndim))

        return dissipation.real

    def vorticity(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate the vorticity of the flow field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        np.ndarray
            Vorticity of the flow field.
        """

        if self.ndim == 3:
            raise ValueError('Vorticity not currently implemented correctly for ndim=3')

        return np.cross(self.nabla, u_hat)


    def fourier_to_phys(self, t_hat: np.ndarray, nref: Optional[int] = None) -> np.ndarray:

        """Transform Fourier domain to physical domain.

        Parameters
        ----------
        t_hat: np.ndarray
            Field tensor in the Fourier domain.
        nref: Optional[int]
            Number of reference points in the physical domain.

        Returns
        -------
        phys: np.ndarray
            Field tensor in the physical domain.
        """

        if not len(t_hat.shape) > self.ndim:
            raise ValueError('Please ensure that a field of the correct shape is passed in.')

        n_leading_dims = t_hat.ndim - (self.ndim + 1)
        leading_dims = t_hat.shape[:n_leading_dims]

        if not nref:
            t_hat_aug = t_hat
            scaling = 1.0
        else:
            ishift = (nref - 2 * self.nk) // 2
            scaling = (nref / self.nk_grid) ** self.ndim

            t_hat_aug = np.zeros(
                ([*leading_dims] + [nref for _ in range(self.ndim)] + [t_hat.shape[-1]]), dtype=np.complex128
            )

            ishift_slice = [slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)]
            t_hat_aug[tuple([...]) + tuple(ishift_slice) + tuple([slice(None)])] = t_hat

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, n_leading_dims + self.ndim

        phys = scaling * np.fft.irfftn(
            np.fft.ifftshift(t_hat_aug, axes=range(axs_lb, axs_ub)),
            s=t_hat_aug.shape[slice(axs_lb, axs_ub)],
            axes=range(axs_lb, axs_ub)
        )

        return phys

    def phys_to_fourier(self, t: np.ndarray) -> np.ndarray:

        """Transform physical domain to Fourier domain.

        Parameters
        ----------
        t: np.ndarray
            Field tensor in the physical domain.

        Returns
        -------
        t_hat: np.ndarray
            Field tensor in the Fourier domain.
        """

        n_leading_dims = t.ndim - (self.ndim + 1)

        nref = t.shape[n_leading_dims]
        ishift = (nref - 2 * self.nk) // 2
        scaling = (self.nk_grid / nref) ** self.ndim

        axs_lb, axs_ub = n_leading_dims, n_leading_dims + self.ndim
        t_hat_padded = scaling * np.fft.fftshift(
            np.fft.fftn(t, s=t.shape[slice(axs_lb, axs_ub)], axes=range(axs_lb, axs_ub)),
            axes=range(axs_lb, axs_ub)
        )

        ishift_slice = [slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)]
        t_hat = t_hat_padded[tuple([...]) + tuple(ishift_slice) + tuple([slice(None)])]

        return t_hat

    def random_field(self, magnitude: float, sigma: float, k_offset: Optional[List[int]] = None) -> np.ndarray:

        """Generate random velocity field in the Fourier domain.

        Parameters
        ----------
        magnitude: float
            Magnitude of the field.
        sigma: float
            Standard deviation of the field.
        k_offset: Optional[List[int]]
            Wavenumber offsets for each dimension.

        Returns
        -------
        u_hat: np.ndarray
            Random field in the Fourier domain.
        """

        if k_offset and len(k_offset) != self.ndim:
            raise ValueError('Must provide offsets for each dimension.')

        random_field = np.random.uniform(size=(*(self.nk_grid for _ in range(self.ndim)), self.ndim))
        delta = np.zeros_like(random_field)

        if k_offset:
            for idx, k in enumerate(k_offset):
                delta[..., idx] -= k

        mag = np.exp(-0.5 * (np.sqrt(np.sum(np.square(self.kt - delta), axis=-1)) / sigma ** 2))
        mag = einops.repeat(mag, '... -> ... b', b=self.ndim)
        mag = magnitude * mag / np.sqrt(2.0 * np.pi * sigma ** 2)

        u_hat = mag * np.exp(2.0j * np.pi * random_field)

        u_hat = np.fft.irfftn(
            np.fft.ifftshift(
                u_hat, axes=range(self.ndim)
            ), s=u_hat.shape[:self.ndim], axes=range(self.ndim)
        )

        u_hat = np.fft.fftshift(
            np.fft.fftn(
                u_hat, s=u_hat.shape[:self.ndim], axes=range(self.ndim)
            ), axes=range(self.ndim)
        )
        u_hat = self.divergence_free(u_hat)

        return u_hat
    
    def divergence_free(self, u_hat: np.ndarray) -> np.ndarray:
        """This function is to make the state field divergence free
        Initially for the to ensure the initial condtions
        The main idea is to formulation a correction term \nabla p_c
        That \nabla. (u+\nabla p_c) = 0"""
        with np.errstate(divide='ignore', invalid='ignore'):
            correction = oe.contract('...u, ...u -> ...', self.kt, -u_hat) / self.kk
            correction[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0
        correction_p = oe.contract('...u, ... -> ...u', self.kt, correction)
        u_hat += correction_p
        return u_hat
        
    def energy_spectrum(self, u_hat: np.ndarray, agg: bool = False) -> np.ndarray:

        """Calculates energy spectrum of flow field.

        Parameters
        ----------
        u_hat: np.ndarray
            Flow-field to calculate energy spectrum of.
        agg: bool
            Determines whether to take a mean.

        Returns
        -------
        ek: np.ndarray
            Energy spectrum of given field.
        """

        n_leading_dims = u_hat.ndim - (self.ndim + 1)
        leading_dims = u_hat.shape[:n_leading_dims]

        uu = 0.5 * oe.contract('...u -> ...', u_hat * np.conj(u_hat)).real
        intk = np.sqrt(self.kk).astype(int)

        ek = np.zeros((tuple([*leading_dims]) + tuple([np.max(intk) + 1])))
        for combo in it.product(*map(range, intk.shape)):
            ek[tuple([...]) + tuple([intk[combo]])] += uu[tuple([...]) + tuple(combo)] / self.nk_grid ** self.ndim

        if agg:
            ek = einops.reduce(ek, '... k -> k', np.mean)

        return ek
