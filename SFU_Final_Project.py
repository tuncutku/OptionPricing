#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:23:12 2019

@author: tuncutku
@SFU - Beedie School of Business
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Import necessary functions and variables
from Simulation import AssetPaths, SRD_generate_paths, random_number_generator_custom
from Calibration import FitJumpDiffusionModel, CIR_calibration
from Calibration import r0, index_returns, S0, f, rho  # Variables
from Valuation import option_pricing
from Quantlib import quantlibprice

"""
Option Specs

"""

# "put" or "call"
option_type = "put"

# Variance reduction techniques: anti_paths or halton
variance_reduction = "anti_paths"

"""
Model Inputs

"""

# Set stock generation parameters
T = 1  # maturity
dt = 1 / 252  # set dt as one day
nPaths = 100  # number of paths
steps = int(T / dt)  # number of steps

"""
Model Calibration
Jumps & Interest Rate

"""

# Calibration of Jump Parameters    
param = np.zeros(5)
param = FitJumpDiffusionModel(index_returns, dt)

# Calibration of Interest Rate Parameters    
kappa_r, theta_r, sigma_r = CIR_calibration()

"""
Simulation

"""

# Create Random Numbers
rand = random_number_generator_custom(nPaths, steps, variance_reduction)

# Simulate interest rates (CIR Model)
r = SRD_generate_paths(r0, kappa_r, theta_r, sigma_r, T, steps, nPaths, rand, rho)

# Simulate Stock Prices 
S = AssetPaths(S0, param, dt, steps, nPaths, rand, rho)

"""
Valuation

"""

# Set Strike Prices
if option_type == "put":

    Market_Data = pd.read_csv("SPDR_Put.csv")

elif option_type == "call":

    Market_Data = pd.read_csv("SPDR_Call.csv")

# Create variable arrays
n_options = Market_Data.shape[0]
price = np.zeros((9, n_options))
times = np.zeros((8, n_options))
price_diff = np.zeros((8, n_options))

# "LS" Least Square
# "NN" Neural Networks 
# "DT" Decision Tree
# "SVM" Support vector machines
# "SGD" Stochastic Gradient Descent
# "ISO" Isotonic Regression
# "GPR" Gaussian Processes Regression
method_type = ["Spot", "Quantlib", "LS", "DT", "SVM", "ISO", "GPR"]
# method_type = ["Spot", "Quantlib", "LS", "NN", "DT", "SVM", "SGD", "ISO", "GPR"]

# Value options with various regresion techniques and strike prices
for idx in range(Market_Data.shape[0]):

    # Set strike price for each option
    K = Market_Data[["Strike"]].iloc[idx, 0]
    market_vol = Market_Data[["Volatility"]].iloc[idx, 0]
    Market_Quote = Market_Data[["Last Price"]].iloc[idx, 0]
    price[0, idx] = Market_Quote

    # Price the option with a built-in function in Quantlib
    start_time = time.time()
    price[1, idx] = quantlibprice(float(K), option_type, float(market_vol))
    price_diff[0, idx] = np.absolute(Market_Quote - price[1, idx])
    end_time = time.time()
    times[0, idx] = end_time - start_time

    # Price the option price with each regression method
    for mdx in range(len(method_type) - 2):
        # Price the option given asset & interest rate paths and other parameters
        start_time = time.time()
        price[mdx + 2, idx] = option_pricing(S, K, r, dt, steps, nPaths, option_type, method_type[mdx + 2])
        end_time = time.time()

        # Calculate time to execute each valuation method
        times[mdx + 1, idx] = end_time - start_time
        # Calculate the difference between market quote and custom prices
        price_diff[mdx + 1, idx] = np.absolute(Market_Quote - price[mdx + 2, idx])

# Calculate average price error and time for each regression method
Average_Errors = np.mean(price_diff, axis=1)
Average_Duration = np.mean(times, axis=1)

"""
Visualization of the interest rate, stock paths and code results

"""
# Plot Spot Rate Curve
plt.plot(f)
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.title("Spot Curve")
plt.show()

# Plot Interest Rates Paths
plt.plot(r)
plt.xlabel('Time')
plt.ylabel('Interest Rates')
plt.title("Interest Rate Paths")
plt.show()

# Plot Stock Prices
plt.plot(S)
plt.xlabel('Time')
plt.ylabel('Index')
plt.title("Index Paths")
plt.show()

# Plot the Distribution of Final Stock Prices
plt.hist(S[-1])
plt.xlabel('Price')
plt.ylabel('Amount')
plt.title("Distribution of Final Stock Prices")
plt.show()

# Plot Average Duration
plt.bar(method_type[1:], Average_Duration)
plt.xlabel('Methods')
plt.ylabel('Duration')
plt.title("Duration by Method")
plt.show()

# Plot Average Errors
plt.bar(method_type[1:], Average_Errors)
plt.xlabel('Methods')
plt.ylabel('Error')
plt.title("Error by Method")
plt.show()

# Plot Calculated Prices
Df_Price = pd.DataFrame(price, index=method_type, columns=Market_Data[["Strike"]].iloc[:, 0])
plt.plot(Df_Price.T)
plt.legend(method_type)
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title("Option Price by Strike Price")
plt.show()
