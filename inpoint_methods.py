import numpy as np
import pandas as pd
import copy as copy
import scipy
import scipy.io
import time
import os
from scipy.linalg import solve, LinAlgWarning
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import openpyxl
import xlsxwriter

#### Determinar el tamaño del paso ####

def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: lightgreen' if is_max.any() else '' for v in is_max]

def paso_intpoint(mu, wmu):
    eps = 0.01
    alphas = np.ones( len(mu) +1 ) # this way the last entry remains one.
    for i in range( len(mu) ):
        if wmu[i] < 0:
            alphas[i] = -(1-eps)*mu[i]/wmu[i]

    alpha = min(alphas)
    return alpha

def paso_intpointJ(mu, wmu):    
    alphas = np.ones((len(wmu)))
    for i in range(len(wmu)):
        if wmu[i] < 0:
            alphas[i] = -mu[i]/wmu[i]
    alpha = min(alphas)
    alfa = min(alpha, 1.0)        
    return alfa
#### Para cargar los matFiles ####

def loadProblem( fname, useSparse = False ):
    mat = scipy.io.loadmat( fname )
    if useSparse:
        A = mat.get('Problem')[0,0][2].astype(float) # sparse matrix
    else:
        A = mat.get('Problem')[0,0][2].astype(float).toarray()
    
    b = mat.get('Problem')[0,0][3].astype(float)[:,0]
    #id = mat.get('Problem')[0,0][4]
    aux = mat.get('Problem')[0,0][5]
    c = aux[0,0][0].astype(float)[:,0]
    print('Norma infinita de b: ', np.linalg.norm(b, np.inf) )
    
    return {'AE':A, 'bE':b, 'c':c}

#### Sistema Completo ####

def intpoint(Q, A, F, c, b, d):
    k = 0
    n = Q.shape[0]
    m = A.shape[0]
    p = F.shape[0]
    
    tol = 10**(-6)
    maxiter = 100
    
    print("El rango de A es", np.linalg.matrix_rank(A))
    
    AT = A.T
    FT = F.T
    
    # Initial values
    x = np.ones(n)
    lamda = np.zeros(m)
    mu = np.ones(p)
    z = F @ x - d + 0.5 * np.ones((p))
    e = np.ones(p)
    tau = (0.5) * (mu.T @ z) / p
    
    # Initial residuals
    v1 = Q @ x + AT @ lamda - FT @ mu + c
    v2 = A @ x - b
    v3 = -F @ x + d + z
    v4 = np.multiply(mu, z)  # Use element-wise product for complementarity condition
    
    ld = np.concatenate((v1, v2, v3, v4), 0)
    norma_cnpo = np.linalg.norm(ld,np.inf)
    
    print("norma_cnpo before iteration",norma_cnpo)
    print("Iter      CNPO        alpha          tau         rcond(K)")
    print('-----------------------------------------------------------')

    warnings.filterwarnings('error', category=LinAlgWarning)
    
    while norma_cnpo > tol and k < maxiter:
        try:
            # Update diagonal matrices Z and U inside the loop
            Z = np.diag(z)
            U = np.diag(mu)
        
            ### KKT Matrix
            row1 = np.hstack((Q, AT, -FT, np.zeros((n, p))))
            row2 = np.hstack((A, np.zeros((m, m + p + p))))
            row3 = np.hstack((-F, np.zeros((p, m + p)), np.identity(p)))
            row4 = np.hstack((np.zeros((p, n + m)), Z, U))
        
            K = np.vstack((row1, row2, row3, row4))
            condK = np.linalg.cond(K,1)
        
            # Perturb the complementarity condition (v4)
            v4_pert = np.multiply(mu, z) - tau * e
            ld_pert = np.concatenate((v1, v2, v3, v4_pert), 0) ### check this
        
            # Solve the linear system with the perturbed residual
            w_vector = scipy.linalg.solve(K, -ld_pert)
       
        except LinAlgWarning:
            print(f"Iteration {k}: Ill-conditioned matrix detected, exiting loop.")
            break
    
        wx = w_vector[0:n]
        wlamda = w_vector[n:n + m]
        wmu = w_vector[n + m:n + m + p]
        wz = w_vector[n + m + p:]
    
        ### Step size
        alpha_mu = paso_intpoint(mu, wmu)
        alpha_z = paso_intpoint(z, wz)
        alpha = 0.995 * min(alpha_mu, alpha_z)
    
        # Update variables
        x += alpha * wx
        mu += alpha * wmu
        lamda += alpha * wlamda
        z += alpha * wz
    
        # Update tau and residuals
        tau = (0.5) * (mu.T @ z) / p
        k += 1
    
        # Recalculate residuals
        v1 = Q @ x + AT @ lamda - FT @ mu + c
        v2 = A @ x - b
        v3 = -F @ x + d + z
        v4 = np.multiply(mu, z)  # Element-wise product
    
        ld = np.concatenate((v1, v2, v3, v4), 0)
        norma_cnpo = np.linalg.norm(ld,np.inf)
    
        print(f"{k:<8}{norma_cnpo:<12.8f}{alpha:<12.8f}{2*tau:<12.8f}{1 / condK:<12.16f}")
    
        if norma_cnpo <= tol or k == maxiter:
            return x, lamda, mu, z, k
    
    return x, lamda, mu, z, k

#### Sistema reducido ####

def intpointR(Q, A, F, c, b, d):
    k = 0
    n = Q.shape[0]
    m = A.shape[0]
    p = F.shape[0]
    
    tol = 10**(-6)
    maxiter = 100
    
    # Initial values
    x = np.ones(n)
    lamda = np.zeros(m)
    mu = np.ones(p)
    z = F @ x - d + 0.5 * np.ones((p))
    tau = (0.5) * (mu.T @ z) / p
    
    print("El rango de A es", np.linalg.matrix_rank(A))
    
    AT = A.T
    FT = F.T
    
    #v1 = Q @ x + AT @ lamda - FT @ mu + c
    #v2 = A @ x - b
    #v3 = -F @ x + d + z
    #v4 = np.multiply(mu, z)  # Element-wise product
    H = np.concatenate((Q @ x + AT @ lamda - FT @ mu + c, A @ x - b, -F @ x + d + z, np.multiply(mu, z)), 0)
    norma_cnpo = np.linalg.norm(H,np.inf)
    print("norma_cnpo before iteration",norma_cnpo)
    print("Iter      CNPO        alpha          tau         rcond(K)")
    print('-----------------------------------------------------------')

    warnings.filterwarnings('error', category=LinAlgWarning)

    while norma_cnpo > tol and k < maxiter:
        try:
            # Update diagonal matrices Z and U inside the loop
            # Initial residuals
            Z = np.diag(z)
            U = np.diag(mu)
            ### KKT Matrix
            D = np.diag(mu / z)
            #print(D)
            G = Q+FT@D@F
            w = np.zeros((p, 1))
            for i in range(p):
                w[i] = F[i, :] @ x - d[i] - (tau / mu[i])
            w = w.ravel()
              
            dg = Q @ x + AT @ lamda - FT@mu + c + FT@D@w
            
            # Define K as a block matrix
            m = A.shape[0]
            K = np.block([
                [G, AT],
                [A, np.zeros((m, m))]
            ])
            
            # Calculate the reciprocal condition number of G
            condK = np.linalg.cond(G,1)
            
            # Define ld
            ld = -np.concatenate([dg, A @ x - b])
            
            # Solve the linear system
            w_vector = scipy.linalg.solve(K, ld)
            
        except LinAlgWarning as e:
            print(f"Iteration {k}: Ill-conditioned matrix detected, exiting loop.")
            print(f"LinAlgWarning details: {e}")
            print(f"mu {mu}")
            print(f"z (holgura) {z}")
            break
    
        wx = w_vector[0:n]
        wlamda = w_vector[n:n + m]
        wmu = -D @ (F @ wx + w)
        wz = -( (1 / mu) * (z * wmu - tau) + z )
        
        ### Step size
        alpha_mu = paso_intpoint(mu, wmu)
        alpha_z = paso_intpoint(z, wz)
        alpha = 0.995 * min(alpha_mu, alpha_z)
        
        # Update variables
        x += alpha * wx
        mu += alpha * wmu
        lamda += alpha * wlamda
        z += alpha * wz
        
        # Update tau and residuals
        tau = (0.5) * (mu.T @ z) / p
        k += 1
        
        H = np.concatenate((Q @ x + AT @ lamda - FT @ mu + c, A @ x - b, -F @ x + d + z, np.multiply(mu, z)), 0)
        norma_cnpo = np.linalg.norm(H,np.inf)
        
        #print("iter=", k, "|", "||cnpo||=", norma_cnpo)
        #print("Condition number of G:", np.linalg.cond(G,1))
        #print("rcond(G)", (1/np.linalg.cond(G,1)))
        #print("alpha", alpha)
        #print("tau =",2*tau)
        print(f"{k:<8}{norma_cnpo:<12.8f}{alpha:<12.8f}{2*tau:<12.8f}{1 / condK:<12.16f}")
    
        if norma_cnpo <= tol or k == maxiter:
            return x, lamda, mu, z, k
        
    return x, lamda, mu, z, k

def intpointR_mask(Q, A, F, c, b, d):
    k = 0
    n = Q.shape[0]
    m = A.shape[0]
    p = F.shape[0]
    
    tol = 1e-6
    kmax = 100
    tau = 0.5 # because mu and z are 1 as well
    
    print("El rango de A es", np.linalg.matrix_rank(A))
    
    AT = A.T
    FT = F.T
    
    # Initial values
    x = np.ones(n)
    lamda = np.zeros(m)
    mu = np.ones(p)
    z = np.ones(p)
    #e = np.ones(p)
    
    v1 = Q @ x + AT @ lamda - FT @ mu + c
    v2 = A @ x - b
    v3 = -F @ x + d + z
    v4 = np.multiply(mu, z)  # Element-wise product
    ld1 = np.concatenate((v1, v2, v3, v4), 0)
    norma_cnpo = np.linalg.norm(ld1,np.inf)

    # Initialize an empty DataFrame to store the iteration results
    highlighted_df = pd.DataFrame(columns=range(Q.shape[0]))  # Q.shape[0] assuming number of rows in the problem


    while norma_cnpo > tol and k < kmax:
        # Update diagonal matrices Z and U inside the loop
        # Initial residuals
        Z = np.diag(z)
        U = np.diag(mu)
        ### KKT Matrix
        row1 = np.hstack((Q, AT, -FT, np.zeros((n, p))))
        row2 = np.hstack((A, np.zeros((m, m + p + p))))
        row3 = np.hstack((-F, np.zeros((p, m + p)), np.identity(p)))
        row4 = np.hstack((np.zeros((p, n + m)), Z, U))

        M = np.vstack((row1, row2, row3, row4))
        
        D = np.diag(mu / z)
        G = G = Q+FT@D@F
        w = np.zeros((p, 1))
        for i in range(p):
            w[i] = F[i, :] @ x - d[i] - (tau / mu[i])
        w = w.ravel()
          
        dg = Q @ x + AT @ lamda - FT@mu + c + FT@D@w
        
        # Define K as a block matrix
        m = A.shape[0]
        K = np.block([
            [G, AT],
            [A, np.zeros((m, m))]
        ])
        
        # Calculate the reciprocal condition number of G
        condK = np.linalg.cond(G,1)
        
        # Define ld
        ld = -np.concatenate([dg, A @ x - b])
        norma_cnpo = np.linalg.norm(ld)
        
        # Solve the linear system
        w_vector = np.linalg.solve(K, ld)
        
        wx     = w_vector[0:n]
        wlamda = w_vector[n:n + m]
        wmu    = -D @ (F @ wx + w)
        wz     = -( (1 / mu) * (z * wmu - tau) + z )
        
        ### Step size
        alpha_mu = paso_intpoint(mu, wmu)
        alpha_z  = paso_intpoint(z, wz)
        #alpha    = 0.995 * min(alpha_mu, alpha_z)
        alpha    = min(alpha_mu, alpha_z)
        print(alpha)
        
        # remember something
        perc_mu = wmu/mu
        perc_z  = wz/z
        
        # Update variables
        x += alpha * wx
        mu += alpha * wmu
        lamda += alpha * wlamda
        z += alpha * wz
        
        # Update tau and residuals
        tau = np.dot(mu, z) / (2 * p)
        k += 1
        
        # Recalculate residuals
        v1 = Q @ x + AT @ lamda - FT @ mu + c
        v2 = A @ x - b
        v3 = -F @ x + d + z
        v4 = np.multiply(mu, z)  # Element-wise product
        
        ld1 = np.concatenate((v1, v2, v3, v4), 0)
        norma_cnpo = np.linalg.norm(ld1,np.inf)
        
        print("\niter=", k, "\t", "||cnpo||=", norma_cnpo)
        print("Condition number of G:", np.linalg.cond(G,1))
        print("rcond(G)", (1/np.linalg.cond(G,1)))
        print("tau =",tau)
        #print(z)
        #print(mu)
        
        mask = mu*z <= 1e-5
        
        #print('cuantos chicos mu*z = %g, vector\n' % (sum(mask)), (mu*z)[mask])
        
        red_mu = []
        if all(mask):
            #neg_mu_mask = (-0.52 < perc_mu) & (perc_mu < -0.48)
            #const_z_mask = (-0.01 < perc_z) & (perc_z < 0.01)
            grow_z_mask = (-0.03 < perc_z) #& (perc_z < 0.01)
            neg_mu = np.arange( len(mask) )[grow_z_mask]
            
            df = pd.DataFrame({'mu': mu, 'pmu': perc_mu, 'z': z, 'pz': perc_z})
            #display( df.style.apply(highlight_greaterthan, threshold=-0.02, column=['pz'], axis=1) )

            highlighted_rows = df[df['pz'] > -0.02].index.tolist()  # Find the rows where pz > -0.02
            print("Rows highlighted in green:", highlighted_rows)

            # Create a row for this iteration (1 for highlighted, 0 for non-highlighted)
            highlighted_row = [1 if i in highlighted_rows else 0 for i in range(Q.shape[0])]

            # Append the current iteration (k) to the DataFrame
            highlighted_df.loc[k] = highlighted_row

            
            #print('mus chicos: vector\n', mu[neg_mu])
            if set(red_mu).issubset( neg_mu ):
                print ('IS subset: GOOD')
            else:
                print ('FAILS subset condition: BAD')
                
            #print('mus chicos: vector\n', neg_mu)
            #print('  change in percentages for mu \n', perc_mu[neg_mu] )
            #print('zs tending to positive contants\n', z[neg_mu] )
            #print('  Largest and smallest change for percentages in entries of z  \n', min(perc_z[neg_mu]), max(perc_z[neg_mu] ))
            red_mu = neg_mu.copy()
        

        if norma_cnpo <= tol or k == kmax:
            display(highlighted_df)

            # Select the last 3 iterations from the DataFrame
            last_3_iterations = highlighted_df.tail(3)

            # Identify columns where all 3 iterations have 1
            highlighted_columns = last_3_iterations.columns[(last_3_iterations == 1).all(axis=0)].tolist()

            # Display the list of highlighted column indexes
            print("Columns highlighted in all of the last 3 iterations:", highlighted_columns)

            
            return x, lamda, mu, z, k
    
    return x, lamda, mu, z, k
