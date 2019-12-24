#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:36:33 2019

@author: tuncutku
"""
import numpy as np
import math

# Functions for random number generation
def HaltonSequence(n,b):
    # This function generates the first n numbers in Halton's low discrepancy sequence with base b

    hs=np.zeros(n)
    
    for idx in range(0,n):
        hs[idx] = Halton_SingleNumber(idx+1,b)
    return hs

def Halton_SingleNumber(n,b):
    # This function shows how to calculate the nth number in Halton's low discrepancy sequence.
    n0 = n
    hn = 0
    f = 1/b

    while n0 > 0:
        
        n1 = math.floor(n0/b)
        r = n0-n1*b
        hn = hn + f*r
        f = f/b
        n0 = n1
    
    return hn

def random_number_generator_custom(nPaths,steps,variance_reduction="anti_paths"):
    
    ''' Function to generate random variables with Halton Sequence

    Parameters
    ==========

    nPaths: float
        Number of Paths
        
    steps: float
        Number of Steps


    Returns
    =======
    matrix: NumPy array
        Random numbers
        
    '''
    if variance_reduction=="anti_paths":
    
        rand = np.random.normal(size=(2,steps,int(nPaths)))
        
    elif variance_reduction=="halton":
    
        rand = np.zeros((2, steps, nPaths), dtype=np.float)

        for jdx in range(0,2):

            # Normally distributed random number generator
            base1 = 2
            base2 = 7
            # generate two Halton sequences
            hs1 = HaltonSequence(int((np.ceil(nPaths*steps)/2)),base1)
            hs2 = HaltonSequence(int(np.ceil((nPaths*steps)/2)),base2)
            # Use the simple Box-Mueller approach to create normal variates
            R = np.sqrt(-2*np.log(hs1))
            Theta = 2*math.pi*hs2
            P = R*np.sin(Theta)
            Q = R*np.cos(Theta)
            # The complete sequence is
            N = np.concatenate((P,Q),axis=0)
            N.resize((steps,nPaths))
            rand[0]=N
    
    return rand

def generate_cholesky(rho_rs):
    ''' Function to generate Cholesky matrix.

    Parameters
    ==========
    rho: float
        correlation between index level and short rate

    Returns
    =======
    matrix: NumPy array
        Cholesky matrix
    '''
    covariance = np.zeros((2, 2), dtype=np.float)
    covariance[0] = [1.0, rho_rs]
    covariance[1] = [rho_rs, 1.0]
    cho_matrix = np.linalg.cholesky(covariance)
    return cho_matrix

def AssetPaths(S0,param,dt,steps,nPaths,rand,rho_rs):
    ''' Function to generate paths for the underlying asset with only jumps
    
    Parameters
    ==========
    S0: float
        initial value of asset
    param: float
        
    dt: float
        long-run mean
    steps: float
        volatility factor
    nPaths: float
        volatility factor
    
    Returns
    =======
    S: NumPy array
        simulated asset paths
    
    '''

    r = param[0]
    sig = param[1]
    alphaJ = param[2]
    sigJ = param[3]
    lambda_1 = param[4]
    
    # Caculate the Drift Factor
    mu = r - sig*sig/2
    muJ = alphaJ - lambda_1*lambda_1/2
        
    # Use the formulation given by MacDonald (2nd Ed. p641)
    k = np.exp(alphaJ)-1
        
    # Create Random Numbers for expNotJ part
    ran = np.zeros((steps, nPaths), dtype=np.float)
    cho_matrix = generate_cholesky(rho_rs)
    
    for t in range(1, steps+1):
        ran[t-1,:] = np.dot(cho_matrix, rand[:, t-1])[1]
    
    # Do the non-jump component
    expNotJ = np.cumprod(np.exp((mu-lambda_1*k)*dt+sig*np.sqrt(dt)*
              ran),axis=0)
        
    # Do the jump component
    m = np.random.poisson(lambda_1*dt,(steps,nPaths))
    expJ = np.exp(m*(muJ+sigJ*np.random.normal(scale=np.maximum(1,np.sqrt(m)),size=(steps,nPaths))))
            
    returns = expNotJ * expJ
        
    # Merge paths with initial stock value
    S = S0*(np.concatenate((np.ones((1,nPaths)),returns),axis=0))

    return S


def SRD_generate_paths(x0, kappa, theta, sigma, T, steps, nPaths, rand, rho_rs):
    ''' Function to simulate Square-Root Difussion (SRD/CIR) process.

    Parameters
    ==========
    x0: float
        initial value
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor
    T: float
        final date/time horizon
    steps: float
        number of time steps
    nPaths: float
        number of paths

    Returns
    =======
    x: NumPy array
        simulated variance paths
    '''
    dt = T / steps
    x = np.zeros((steps, nPaths), dtype=np.float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0
    sdt = np.sqrt(dt)
    cho_matrix = generate_cholesky(rho_rs)
    
    for t in range(1, steps):
        
        ran = np.dot(cho_matrix, rand[:, t-1])

        xh[t] = (xh[t - 1] + kappa * (theta - np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[0] * sdt)
        x[t] = np.maximum(0, xh[t])

    return x