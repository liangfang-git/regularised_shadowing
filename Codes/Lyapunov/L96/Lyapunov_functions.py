import numpy as np
from scipy.linalg import block_diag
from decimal import Decimal, getcontext

class Lyapunov(object):

    def __init__(Lyapunov, n_lyap, stept, t_forward_tran, t_forward_develop, t_back_tran, ioutp):
        Lyapunov.n_lyap = n_lyap
        Lyapunov.stept = stept
        Lyapunov.tf_t = t_forward_tran
        Lyapunov.tf_d = t_forward_develop
        Lyapunov.tb_t = t_back_tran
        Lyapunov.tend = t_forward_tran+t_forward_develop
        Lyapunov.ioutp = ioutp

    def Lyapunov_QR(Lyapunov, Sim, u0):
        factor = 1
        tstart = 0
        n1 = Sim.nx  # n1 = number of non-linear ODEs
        n2 = n1 * (factor*Lyapunov.n_lyap + 1)  # n2 = n*(n+1) = total number of ODEs

        # Number of iterations
        nit = int(round((Lyapunov.tf_t - tstart) / Lyapunov.stept))

        # Memory allocation
        w = np.zeros(n2)  # pre-allocation for the tangent vectors
        cum = np.zeros(Lyapunov.n_lyap)  # pre-allocation for the Lyapunov exponents

        # Initial values
        Z = np.random.rand(factor*Sim.nx,Lyapunov.n_lyap)
        Qz, _ = np.linalg.qr(Z)
        # Qz = np.eye(factor*Sim.nx)[:,:Lyapunov.n_lyap]
        w[n1:n2] = Qz.flatten()
        if (u0.any()):
            y1 = u0
        else:
            y1 = Sim.precursor_simulation()
            # Allowing the trajectory to develop and get into the attractor
        w[:n1] = y1

        t = tstart

        # Output modification
        Lyapunovxp = []
        Texp = []

        # Main loop
        for ITERLYAP in range(1, nit + 1):
            # Solution of extended ODE system
            w1 = Sim.LE_multiple_linear_simulation(round(t,3), round(t+Lyapunov.stept,3), w, Lyapunov.n_lyap)
            t += Lyapunov.stept
            w = w1.copy()
            x = w[n1:n2]  # extract vector for QR factorization
            YY = x.reshape(factor*Sim.nx,Lyapunov.n_lyap) # change vector to matrix for QR
            Q, R = np.linalg.qr(YY)  # QR factorization of updated w values
            # n_lyap is the number of exponents to be calculated, change accordingly for different systems
            for k in range(Lyapunov.n_lyap):
                cum[k] += np.log(np.abs(R[k, k]))
                
            w[n1:n2] = Q.ravel()  # the Q matrix provides the new tangent vectors for the next iteration
            # Normalize exponent with time
            lp = cum / (t - tstart)

            # Output modification
            if ITERLYAP == 1:
                Lyapunovxp = lp
                Texp = t
            else:
                Lyapunovxp = np.vstack((Lyapunovxp, lp))
                Texp = np.append(Texp, t)

            if ITERLYAP % Lyapunov.ioutp == 0:  # if remainder after ITERLYAP/ioutp = 0 then print result
                print(f"t={t:.4f}", end="")
                for val in lp:
                    print(f" {val:10.6f}", end="")
                print()

        lambda_QR = np.real(Lyapunovxp[-1, :])

        # We continue but now Q and R are stored to compute CLV later
        Phi_list, R_list1 = [Q], []

        tstart = 0.0
        t = tstart
        nit0 = int(round((Lyapunov.tf_d - Lyapunov.tb_t - tstart) / Lyapunov.stept))
        w_out = np.zeros((nit0+1,n1))
        w_out[0,:] = w[:n1].copy()
        for ITERLYAP in range(nit0):
            w1 = Sim.LE_multiple_linear_simulation(round(t,3), round(t+Lyapunov.stept,3), w, Lyapunov.n_lyap)
            t += Lyapunov.stept
            w = w1.copy()
            w_out[ITERLYAP+1,:] = w[:n1].copy()
            x = w[n1:n2]  # extract vector for QR factorization
            YY = x.reshape(factor*Sim.nx,Lyapunov.n_lyap) # change vector to matrix for QR
            Q, R = np.linalg.qr(YY)  # QR factorization of updated w values
            w[n1:n2] = Q.ravel()  # the Q matrix provides the new tangent vectors for the next iteration
            Phi_list.append(Q)
            R_list1.append(R)

        # Now we only store R to compute A- later
        R_list2 = []
        tstart = 0.0
        t = tstart
        nit = int(round((Lyapunov.tb_t-tstart) / Lyapunov.stept))
        for _ in range(nit):
            w1 = Sim.LE_multiple_linear_simulation(round(t,3), round(t+Lyapunov.stept,3), w, Lyapunov.n_lyap)
            t += Lyapunov.stept
            w = w1.copy()
            x = w[n1:n2]  # extract vector for QR factorization
            YY = x.reshape(factor*Sim.nx,Lyapunov.n_lyap) # change vector to matrix for QR
            Q, R = np.linalg.qr(YY)  # QR factorization of updated w values
            w[n1:n2] = Q.ravel()  # the Q matrix provides the new tangent vectors for the next iteration
            R_list2.append(R)
        
        # Generate A make it converge to A-
        A = np.triu(np.random.rand(Lyapunov.n_lyap,Lyapunov.n_lyap))
        for R in reversed(R_list2):
            C = np.diag(1. / np.linalg.norm(A, axis = 0))
            B = A @ C
            A = np.linalg.solve(R, B)
        del R_list2

        # Compute CLV at checkpoints
        CLV = [Phi_list[-1] @ A]
        for Q, R in zip(reversed(Phi_list[:-1]), reversed(R_list1)):
            C = np.diag(1. / np.linalg.norm(A, axis = 0))
            B = A @ C
            A = np.linalg.solve(R, B)
            CLV_t = Q @ A
            CLV.append(CLV_t / np.linalg.norm(CLV_t, axis = 0))

        CLV.reverse()
        CLV = np.array(CLV)
        
        
        del R_list1
        del Phi_list  
              

        return lambda_QR,Lyapunovxp,Texp,CLV,w_out   

