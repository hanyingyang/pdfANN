# ANN_pipeline/func
from __future__ import annotations
from scipy.stats import beta
from scipy import interpolate
from typing import Optional, Union

import os

import numpy as np
import numpy.matlib as npm

def KLDiv(P, Q):
    """
    dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
    distributions
    P and Q  are automatically normalized to have the sum of one on rows
    have the length of one at each 
    P =  n x nbins
    Q =  1 x nbins or n x nbins(one to one)
    dist = n x 1
    """
    P = P + 1e-16
    Q = Q + 1e-16
    if np.size(P, 1) != np.size(Q, 1):
        print('The number of columns in P and Q shoulb be the same')
    #normalizing P and Q
    if np.size(Q, 0) == 1:
        Q = Q / np.sum(Q)
        P = P / npm.repmat(np.sum(P, 1), 1, np.size(P, 1))
        dist = np.sum(P * np.log(P / npm.repmat(Q, np.size(P, 1), 1)), axis = 1)

    elif np.size(Q, 0) == np.size(P, 0):
        Q = Q / npm.repmat(np.sum(Q, 1), 1, np.size(Q, 1))
        P = P / npm.repmat(np.sum(P, 1), 1, np.size(P, 1))
        dist = np.sum(P * np.log(P / Q), axis=1)
    
    np.nan_to_num(dist, nan=0)
    
    return(dist)

def JSDiv(P, Q):
    """
    Jensen-Shannon divergence of two probability distributions
    dist = JSD(P,Q) Kullback-Leibler divergence of two discrete probability
    distributions
    P and Q  are automatically normalized to have the sum of one on rows
    have the length of one at each 
    P =  n x nbins
    Q =  1 x nbins
    dist = n x 1
    """
    if np.size(P, 1) != np.size(Q, 1):
        print('The number of columns in P and Q shoulb be the same')
        
    Q = Q / np.sum(Q)
    Q = npm.repmat(Q, np.size(P, 0), 1)
    P = P / (npm.repmat(np.sum(P, 1), 1, np.size(P, 1)))
    
    M = 0.5 * (P + Q)
    return((0.5 * KLDiv(P, M)) + (0.5 * KLDiv(Q, M)))

def copulaPDF(i, j, k, centres, edges, input_zc):
    cbar = input_zc[0]
    gc = input_zc[3] / (cbar * (1 - cbar) + 1e-8)
    aa = cbar * (1 / gc - 1) 
    bb = (1 - cbar) * (1 / gc - 1)
    c_space = np.linspace(0, 1, 101)
    betaPDF = beta.pdf(c_space, aa, bb)
    if sum(np.isinf(betaPDF)) > 0:
        betaPDF[np.isinf(betaPDF).nonzero()] = 0.
    betaPDF = betaPDF / np.trapz(betaPDF, x=c_space)
    betaCDF = np.zeros(betaPDF.shape)
    betaCDF[0] = betaPDF[0]
    for ii in range(1,len(c_space)):
        betaCDF[ii] = np.trapz(betaPDF[0:ii], x=c_space[0:ii])

    zbar = input_zc[1]
    gz = input_zc[2] / (zbar * (1 - zbar) + 1e-8)
    aa = zbar * (1 / gz - 1) 
    bb = (1 - zbar) * (1 / gz - 1)
    z_space = np.logspace(-3, np.log10(max(edges[1]) / 2), 98)
    z_space = np.insert(z_space, 0, [0.], axis=0)
    z_space = np.insert(z_space, len(z_space), [max(edges[1])], axis=0)
    zbetaPDF = beta.pdf(z_space, aa, bb)
    if sum(np.isinf(zbetaPDF)) > 0:
        zbetaPDF[np.isinf(zbetaPDF).nonzero()] = 0.
    zbetaPDF = zbetaPDF / np.trapz(zbetaPDF, x=z_space)
    zbetaCDF = np.zeros(zbetaPDF.shape)
    zbetaCDF[0] = zbetaPDF[0]
    for ii in range(1,len(z_space)):
        zbetaCDF[ii] = np.trapz(zbetaPDF[0:ii], x=z_space[0:ii])
    
    cov = input_zc[4]
    tmp = np.array([gz * zbar * (1 - zbar), gc * cbar * (1 - cbar), cov, max(edges[1])])
    cplDir = 'copula_v3'
    with open('copulaDir.txt','w') as strfile:
        strfile.write(cplDir)
    np.savetxt(cplDir + '/copula_data.txt',tmp[np.newaxis,:],
               fmt='%.5e %.5e %.5e %.5e')
    np.savetxt(cplDir + '/betaPDFs.txt',
                np.concatenate((betaPDF[np.newaxis,:], zbetaPDF[np.newaxis,:],
                                betaCDF[np.newaxis,:], zbetaCDF[np.newaxis,:])
                              ,axis=1).T, fmt='%.5e')
    os.system(cplDir + '/copula')
    fln = (cplDir + '/copulaJPDF_.dat')
    data = np.loadtxt(fln)
    cplPDF = data.reshape(len(c_space), len(z_space), order='F')
    f = interpolate.interp2d(z_space, c_space, cplPDF)
    cplPDF_intp = f(centres[1], centres[0])

    return cplPDF_intp

def betaPDF(mean: np.ndarray, g: np.ndarray, center: np.ndarray, width: np.ndarray) -> np.ndarray:
    mean   = np.asarray(mean,   dtype=float)
    g      = np.asarray(g,      dtype=float)
    center = np.asarray(center, dtype=float)
    width  = np.asarray(width,  dtype=float)

    # ---- Check validity of inputs -----------------------------------------
    if np.any(g < 0):
        raise ValueError("Variance must be >= 0 for a Beta/Dirac model.")

    N = mean.size
    M = center.size

    pdf = np.zeros((N, M), dtype=float)

    # Identify degenerate (Dirac) vs non-degenerate (Beta) cases
    # Threshold: if std dev < thr * dx_min, treat as Dirac
    thr = 0.05  
    sigma = np.sqrt(g * (mean * (1 - mean)))  # standard devariation
    dx_min = np.min(width)
    
    degenerate = (g == 0.0) | (sigma < thr * dx_min)
    nondeg_idx = np.where(~degenerate)[0]

    # ---- Handle non-degenerate cases: Beta distributions ------------
    if nondeg_idx.size > 0:
        mean_nd = mean[nondeg_idx].copy()
        g_nd  = g[nondeg_idx].copy()
    
        eps_mean = 1e-6
        mean_nd = np.clip(mean_nd, eps_mean, 1.0 - eps_mean)

        eps_g = 1e-12
        if np.any(g_nd <= eps_g) or np.any(g_nd >= 1 - eps_g):
            raise ValueError("All g values must satisfy 0 < g < 1 for valid beta parameters.")
        
        ab_sum = 1.0 / g_nd - 1.0                      
        ab_sum = np.maximum(ab_sum, 1e-12)          # numerical safety

        a = mean_nd * ab_sum
        b = (1.0 - mean_nd) * ab_sum

        eps_x = 1e-12
        center_safe = np.clip(center, eps_x, 1.0 - eps_x)

        beta_pdf_nd = beta.pdf(center_safe, a[:, None], b[:, None])
        beta_pdf_nd[~np.isfinite(beta_pdf_nd)] = 0.0

        pdf[nondeg_idx, :] = beta_pdf_nd

    # ---- Handle degenerate cases: Dirac delta at mean[i] -----------------
    deg_idx = np.where(degenerate)[0]
    for i in deg_idx:
        # Find the grid point closest to mean[i]
        idx = np.argmin(np.abs(center - mean[i]))
        # All zeros except one spike
        if width[idx] <= 0:
            raise ValueError("Width values must be positive to define integrals.")
        pdf[i, :] = 0.0
        pdf[i, idx] = 1.0 / width[idx]

    # ---- Normalise each row so that integral over the grid ≈ 1 -----------
    weighted = pdf * width  
    int_pdf = np.sum(weighted, axis=1, keepdims=True) 

    if np.any(int_pdf <= 0):
        raise RuntimeError("Zero or negative integral during normalisation.")

    return pdf / int_pdf
