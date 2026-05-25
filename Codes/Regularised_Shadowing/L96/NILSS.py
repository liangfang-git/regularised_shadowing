import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import svdvals
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, splev, splder
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 7,
          'figure.figsize': (3, 3),
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

def find_corner_fitting(rho,eta,reg_param,sigma,uy,plot_curve,r,c):
    drho = np.copy(rho)
    ddrho = np.copy(rho)
    deta = np.copy(eta)
    ddeta = np.copy(eta)
    drho_hat = np.copy(rho)
    ddrho_hat = np.copy(rho)
    deta_hat = np.copy(eta)
    ddeta_hat = np.copy(eta)
    
    kappa = np.copy(eta)
    
    n = len(sigma)
    
    for i in range(len(rho)):
       deta[i] = 0.0
       ddeta[i] = 0.0
       drho[i] = 0.0
       ddrho[i] = 0.0
       for j in range(n):
           deta[i] += 4*reg_param[i]**3*sigma[j]/((reg_param[i]**2+sigma[j])**3)*(uy[j])**2
           ddeta[i] += 12*reg_param[i]**2*sigma[j]*(sigma[j]-reg_param[i]**2)/((reg_param[i]**2+sigma[j])**4)*(uy[j])**2
           drho[i] -= 4*reg_param[i]*sigma[j]/((reg_param[i]**2+sigma[j])**3)*(uy[j])**2
           ddrho[i] += 4*sigma[j]*(5*reg_param[i]**2-sigma[j])/((reg_param[i]**2+sigma[j])**4)*(uy[j])**2
       
       deta_hat[i] = deta[i]/eta[i]
       drho_hat[i] = drho[i]/rho[i]
       
       ddeta_hat[i] = (ddeta[i]*eta[i]-deta[i]**2)/(eta[i]**2)
       ddrho_hat[i] = (ddrho[i]*rho[i]-drho[i]**2)/(rho[i]**2)
           
       kappa[i] = -2*(drho_hat[i] * ddeta_hat[i] - ddrho_hat[i] * deta_hat[i]) / ((drho_hat[i]**2 + deta_hat[i]**2)**(1.5))

    kmax = np.max(kappa)
    ikmax = np.argmax(kappa)
    if (kmax < 0):
        ikmax = 0
        print ("Not regularised")
    reg_c = reg_param[ikmax]
    rho_c, eta_c = rho[ikmax], eta[ikmax]
    if plot_curve:
        np.savez(f"./l_curve_plot/{int(c)}/l_curve_{r}",sigma = sigma, kappa = kappa, kmax = kmax, rho = rho, eta = eta, rho_c=rho_c, eta_c = eta_c)
    return reg_param[ikmax],reg_c
    
def plot_lcurve(rho,eta,rho_c,eta_c,r,c):
    plt.figure()
    plt.plot(rho,eta,'-',markersize=2,lw=1)
    plt.plot(rho_c,eta_c,'o')
    plt.xscale("log",base=10)
    plt.yscale("log",base=10)
    plt.xlabel(r"$\|\rho(\gamma)\|_2^2$")
    plt.ylabel(r"$\|\eta(\gamma)\|_2^2$")
    plt.grid()
    plt.savefig(f"./l_curve_plot/{int(c)}/rho_eta_F_{int(c)}_r_{r}.pdf")

class NILSS(object):

    def __init__(NILSS, n_ple, dT, nseg, design_method):
        NILSS.n_ple = n_ple
        NILSS.dT = dT
        NILSS.nseg = nseg
        NILSS.nstep = 0
        NILSS.design = design_method

    def pushSeg(NILSS, Sim, x0, factor):
        """
        find u, w, vstar, f, J, dJdu on each segment.
        Here we store all values at all time steps for better intuition,
        but this implementation costs a lot of memory.
        See the paper on FD-NILSS for discussion on how to reduce memory cost 
        by computing inner products via only snapshots.
        """
        t_sim = np.zeros((NILSS.nseg, NILSS.nstep))
        u = np.zeros((NILSS.nseg, NILSS.nstep, Sim.nx))
        f = np.zeros(u.shape)
        vstar = np.zeros([NILSS.nseg, NILSS.nstep, factor*Sim.nx])
        vstar_perp = np.zeros(vstar.shape)
        w = np.zeros([NILSS.nseg, NILSS.nstep, factor*Sim.nx, NILSS.n_ple])
        w_perp = np.zeros(w.shape)
        Rs = [] #Rs[0] in code = R_1 in paper
        bs = [] #bs[0] in code = b_1 in paper

        # push forward
        for iseg in range(0, NILSS.nseg):
            # compute u, w, vstar, f, J
            x1, t_sim[iseg,:] = Sim.NILSS_forward(iseg*NILSS.dT, (iseg+1)*NILSS.dT, x0, NILSS.nstep, NILSS.n_ple)
            u[iseg,:,:] = x1[:,:Sim.nx]
            for i in range(0, NILSS.nstep):
                w[iseg,i,:,:] = x1[i,Sim.nx:(factor*NILSS.n_ple+1)*Sim.nx].reshape(factor*Sim.nx, NILSS.n_ple)
            vstar[iseg,:,:] = x1[:,(factor*NILSS.n_ple+1)*Sim.nx:]
            for istep in range(0, NILSS.nstep):
                f[iseg, istep] = Sim.rhs_calculate(u[iseg, istep])
            # calculate vstar_perp and w_perp
            vstar_perp = vstar.copy()
            w_perp = w.copy()
            for i in range(0, NILSS.nstep):
                vstar_perp[iseg, i, :Sim.nx] = vstar[iseg, i, :Sim.nx] - np.dot(vstar[iseg, i, :Sim.nx], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]
                for ius in range(0, NILSS.n_ple):
                    w_perp[iseg, i, :Sim.nx, ius] = w[iseg, i, :Sim.nx, ius] - np.dot(w[iseg, i, :Sim.nx, ius], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]
            Q_temp, R_temp = np.linalg.qr(w_perp[iseg,-1], 'reduced')

            Rs.append(R_temp)
            b_temp = Q_temp.T @ vstar_perp[iseg,-1]
            bs.append(b_temp)
            p_temp = vstar_perp[iseg,-1] - Q_temp @ b_temp

            if iseg < NILSS.nseg - 1:
                u[iseg+1, 0] = u[iseg, -1]
                w[iseg+1, 0] = Q_temp.copy()
                vstar[iseg+1,0] = p_temp.copy()
            x0[:Sim.nx] = u[iseg, -1]
            x0[Sim.nx:(factor*NILSS.n_ple+1)*Sim.nx] = Q_temp.flatten()
            x0[(factor*NILSS.n_ple+1)*Sim.nx:] = p_temp[:]

        return [t_sim, u, w, vstar, w_perp, vstar_perp, f[:,-1,:], Rs[:-1], bs[:-1]]

    def NILSS_v_calculation(NILSS, Sim, u0, w_rand, regularised,r,c,plot_curve):
        factor = 1
        NILSS.nstep = int(round(NILSS.dT / Sim.dt)) + 1 # number of step + 1 in each time segment
        # push forward u to a stable attractor
        w0 = np.zeros((factor*Sim.nx,NILSS.n_ple))
        w_perb = np.zeros((factor*Sim.nx,NILSS.n_ple))

        for ius in range (NILSS.n_ple):
            w0[:,ius] = w_rand[:,ius]#np.random.rand(factor*Sim.nx)
            w0[:,ius] = w0[:,ius] / np.linalg.norm(w0[:,ius])
        if (u0.any()):
            u_ps = u0.copy()
        else:
            u_ps = Sim.precursor_simulation()
        f_temp = Sim.rhs_calculate(u_ps)
        w_perb = w0.copy()
        for ius in range(0, NILSS.n_ple):
            w_perb[:Sim.nx,ius] = w0[:Sim.nx,ius] - np.dot(w0[:Sim.nx,ius], f_temp) / np.dot(f_temp, f_temp) * f_temp
        Q_temp, _ = np.linalg.qr(w_perb, 'reduced')
        w0 = Q_temp.copy()

        xn0 = np.zeros(Sim.nx*(factor*NILSS.n_ple+1+factor))
        xn0[:Sim.nx] = u_ps.copy()
        xn0[Sim.nx:(factor*NILSS.n_ple+1)*Sim.nx] = w0.flatten()

        t_sim, u, w, vstar, w_perp, vstar_perp, f_ksi, Rs, bs = NILSS.pushSeg(Sim, xn0, factor)

        # Construct Schur complement of the Lagrange multiplier method of the NILSS problem.
        # See the paper on FD-NILSS for this neat method
        # find C^-1

        Cinvs = []
        for iseg in range(NILSS.nseg):
            if (NILSS.design == 'CheckPoint'):
                Cinvs.append(np.eye(NILSS.n_ple))
            elif (NILSS.design == 'FullTrajectory'):
                C_iseg = np.zeros((NILSS.n_ple, NILSS.n_ple))
                for i in range(NILSS.n_ple):
                    for j in range(NILSS.n_ple):
                        C_iseg[i,j] = np.sum(np.trapz(w_perp[iseg, :, :Sim.nx, i] * w_perp[iseg, :, :Sim.nx, j], x=t_sim[iseg], axis=0))
                Cinvs.append(np.linalg.inv(C_iseg))
            else:
                print ("Only works for CheckPoint and FullTrajectory design.")
        Cinv = block_diag(*Cinvs)

        # construct d
        ds = []
        for iseg in range(NILSS.nseg):
            d_iseg = np.zeros(NILSS.n_ple)
            if (NILSS.design == 'Checkpoint'):
                for i in range(NILSS.n_ple):
                    d_iseg[i] = np.sum(w_perp[iseg, 0, :Sim.nx, i] * vstar_perp[iseg, 0, :Sim.nx])
            elif (NILSS.design == 'FullTrajectory'):
                for i in range(NILSS.n_ple):
                    d_iseg[i] = np.sum(np.trapz(w_perp[iseg, :, :Sim.nx, i] * vstar_perp[iseg, :, :Sim.nx], x=t_sim[iseg], axis=0))

            ds.append(d_iseg)
        d = np.ravel(ds) 

        # construct B, first off diagonal I, then add Rs
        B = np.eye((NILSS.nseg-1)*NILSS.n_ple, NILSS.nseg*NILSS.n_ple, k=NILSS.n_ple)
        B[:, :-1*NILSS.n_ple] -= block_diag(*Rs)

        # construct b
        b = np.ravel(bs)

        vsquare = 0

        for iseg in range(NILSS.nseg):
            if (NILSS.design == 'CheckPoint'):
                vsquare += np.sum(vstar_perp[iseg, 0, :Sim.nx] * vstar_perp[iseg, 0, :Sim.nx])
 
            if (NILSS.design == 'FullTrajectory'):
                vsquare += np.sum(np.trapz(vstar_perp[iseg, :, :Sim.nx] * vstar_perp[iseg, :, :Sim.nx], x=t_sim[iseg], axis=0))#/(NILSS.nseg*NILSS.dT)

        a = NILSS.solve_matrix_schur(Cinv, B, b, d, vsquare, regularised,r,c,plot_curve)

        # calculate v and vperp
        v = np.zeros((NILSS.nseg, NILSS.nstep, Sim.nx))
        for iseg in range(NILSS.nseg):
            v[iseg] = vstar[iseg,:,:Sim.nx]
            for ius in range(0, NILSS.n_ple):
                v[iseg] += a[iseg, ius] * w[iseg,:,:Sim.nx,ius]        

        # calculate ksi, only need to use last step in each segment
        ksi = np.zeros(NILSS.nseg)
        for iseg in range(NILSS.nseg):
            ksi[iseg] = np.dot(v[iseg, -1], f_ksi[iseg]) / np.dot(f_ksi[iseg], f_ksi[iseg])

        return u, v, t_sim, ksi 
    
        # Calculate the sensitivity
    def NILSS_sensitivity(NILSS, u, v, t_sim, ksi, JJu_space):
        J_seg = np.zeros(NILSS.nseg)
        dJ_seg = np.zeros(NILSS.nseg)
        Jt = np.zeros((NILSS.nseg,NILSS.nstep))
        dJt = np.zeros((NILSS.nseg,NILSS.nstep))
        for iseg in range(NILSS.nseg):
            for istep in range(NILSS.nstep):
                Jt[iseg, istep], dJt[iseg, istep] = JJu_space(u[iseg,istep],v[iseg,istep])
            J_seg[iseg] = 1.0/(NILSS.dT*NILSS.nseg)*np.trapz(Jt[iseg], x=t_sim[iseg], axis=0)
        J = np.sum(J_seg)
        for iseg in range(NILSS.nseg):
            dJ_seg[iseg] = 1.0/(NILSS.dT*NILSS.nseg)*(np.trapz(dJt[iseg],x=t_sim[iseg],axis=0) + ksi[iseg] * (J - Jt[iseg,-1]))
        dJ = np.sum(dJ_seg)
        return J, dJ
    
    def solve_matrix_schur(NILSS, Cinv, B, b, d, vsquare, regularised,r,c,plot_curve):
        if regularised == 0:
            reg_c = 0.0
        else:
            npoints = 100
            smin_ratio = 16*np.finfo(float).eps
            eta = np.zeros(npoints)
            rho = np.zeros(npoints)           
            reg_param = np.zeros(npoints)

            U, sigma, Vt = np.linalg.svd(B @ Cinv @ B.T, full_matrices=False)

            S = 1*(sigma)**0.5
            
            reg_param[-1] = max(S[-1],S[0]*smin_ratio)
            ratio = (S[0]/reg_param[-1])**(1./(npoints-1))
            for i in range(1,npoints): 
                reg_param[-1*i-1] = ratio*reg_param[-i]
            reg_param = reg_param[::-1]
            b1 = -B @ Cinv @ d - b
            uy_temp = np.dot(U.T, b1)
            
            for i in range(len(reg_param)):
                for j in range(len(sigma)):
                    eta[i] += reg_param[i]**4*(1.0/(reg_param[i]**2+sigma[j])*uy_temp[j])**2
                for j in range(len(sigma)):
                    rho[i] += sigma[j]*(1.0/(reg_param[i]**2+sigma[j])*uy_temp[j])**2
            reg_c, _ = find_corner_fitting(rho,eta,reg_param,sigma,uy_temp,plot_curve,r,c)
            print (reg_c,reg_param[0],reg_param[-1])
        lbd = np.linalg.solve(-B @ Cinv @ B.T- reg_c**2* np.eye(len(B)), B @ Cinv @ d + b)
        a = -Cinv @ (B.T @ lbd + d)
        if plot_curve:
            eta_c = 0
            rho_c = 0
            for j in range(len(sigma)):
                eta_c += reg_c**4*(1.0/(reg_c**2+sigma[j])*uy_temp[j])**2
                rho_c += sigma[j]*(1.0/(reg_c**2+sigma[j])*uy_temp[j])**2
            plot_lcurve(rho,eta,rho_c,eta_c,r,c)
        a = a.reshape([NILSS.nseg, NILSS.n_ple])
        return a

