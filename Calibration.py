#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:34:32 2019

@author: tuncutku
"""

# Import Libraries
import numpy as np
import scipy.interpolate as sci
import math
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import fmin

"""
Input for Calibrations
"""

# Input for CIR Calibrations
# Source for OIS: https://www.global-rates.com/interest-rates/libor/american-dollar/usd-libor-interest-rate-overnight.aspx
# Source for zero coupon bonds: https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield
rate_data = pd.read_csv("zcb.csv", parse_dates=['date'], infer_datetime_format = True)
r_list = rate_data.iloc[0,1:]/100
t_list = np.array((1, 7, 30, 90, 180, 360, 720)) / 365

zero_rates = r_list
r0 = r_list.iloc[0] # 0.0  # set to zero


tck = sci.splrep(t_list, zero_rates)  # cubic splines
tn_list = np.linspace(0.0, 1, 3*len(r_list))
ts_list = sci.splev(tn_list, tck, der=0)
de_list = sci.splev(tn_list, tck, der=1)

f = ts_list + de_list * tn_list

# Import S&P 500 SPDR Values
SPY = pd.read_csv("SPY.csv")
index_returns = (np.log(SPY.Close)-np.log(SPY.Close.shift(1)))[1:]
S0 = SPY.Close.iloc[-1]

"""
Correlation Between Index and overnight LIBOR
"""
# Pull overnight LIBOR data
# Source: https://fred.stlouisfed.org/series/USDONTD156N
libor = pd.read_csv("LIBOR.csv")
libor_returns = (np.log(libor.Rate)-np.log(libor.Rate.shift(1)))[1:]
rho = np.corrcoef(libor_returns, index_returns)[0,1]

"""
Jump Calibration
"""

# Calibration of Jump Diffusion Model
def FitJumpDiffusionModel(X,dt):
    
    # Determine initial estimates for the model parameters
    x0 = np.zeros(5) # These are [alpha;sig;alphaJ;sigJ;lambda]
    
    # start alpha and sig at their values if there are no jumps
    x0[0] = np.mean(X)/dt # mu
    x0[1] = np.std(X) # sig
    
    # alphaJ and sigJ are intialized from the returns
    x0[2] = x0[0] # alphaJ
    x0[3] = x0[1] # sigJ
    x0[4] = 0 # lambda
    
    # Perform an optimization to get the model parameters
    
    boundary=[[-math.inf,math.inf],[np.finfo(float).eps,math.inf],
            [-math.inf,math.inf],[np.finfo(float).eps,math.inf],
            [np.finfo(float).eps,1-np.finfo(float).eps]]
    
    llf = lambda params:localMaximumLikelihoodCostFcn(params,X,dt)
    
    params = minimize(llf,x0,bounds=boundary).x
    
    return params

def localMaximumLikelihoodCostFcn(param,X,dt):
# Maximum likelihood cost function for the mean revision jump diffusion model

    # To help with the explanation, break out the parameters
    r=param[0]
    sig=param[1]
    alphaJ = param[2]
    sigJ = param[3]
    lambda_1 = param[4]

    mu = r - sig*sig/2
    muJ = alphaJ - lambda_1*lambda_1/2
    

    # Use the formulation given by MacDonald (2nd Ed. p641)
    k = np.exp(alphaJ)-1

    cost = -np.sum(np.log(lambda_1*norm.pdf(X,(mu-lambda_1*k)*dt+muJ,
           np.sqrt(sig**2*dt+sigJ**2))+(1-lambda_1)*norm.pdf(X,(mu-lambda_1*k)*dt,
           sig*np.sqrt(dt))))
    
    return cost

"""
Interest Rate Calibration
"""

def CIR_forward_rate(opt):
    ''' Function for forward rates in CIR85 model.

    Parameters
    ==========
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean
    sigma_r: float
        volatility factor

    Returns
    =======
    forward_rate: float
        forward rate
    '''
    
    kappa_r, theta_r, sigma_r = opt
    t = tn_list
    g = np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)
    sum1 = ((kappa_r * theta_r * (np.exp(g * t) - 1)) /
            (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)))
    sum2 = r0 * ((4 * g ** 2 * np.exp(g * t)) /
                 (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)) ** 2)
    forward_rate = sum1 + sum2
    return forward_rate


def CIR_error_function(opt):
    ''' Error function for CIR85 model calibration. '''
    
    kappa_r, theta_r, sigma_r = opt
    if 2 * kappa_r * theta_r < sigma_r ** 2:
        return 100
    if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
        return 100
    forward_rates = CIR_forward_rate(opt)
    MSE = np.sum((f - forward_rates) ** 2) / len(f)
    # print opt, MSE
    return MSE

def CIR_calibration():
    opt = fmin(CIR_error_function, [0.3, 0.04, 0.1],
               xtol=0.00001, ftol=0.00001,
               maxiter=500, maxfun=1000)
    return opt