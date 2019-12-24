#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:30:16 2019

@author: tuncutku
"""
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm,linear_model
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor

def option_pricing(S,K,r,dt,steps,nPaths,option_type,method_type):
    
    # Calculate intrinsic values
    if option_type=="put":
        cFlows = np.maximum(K-S[1:,:],0)
    elif option_type=="call":
        cFlows = np.maximum(S[1:,:]-K,0)
            
    # Calculate the discount factor for each time step
    disc = np.exp(-r*dt)    
    
    # Loop backwards in time filling the cash flows matrix
    
    for idx in range(steps-2,-1,-1):

        # Determine which cashflows to regress
        mask = cFlows[idx,:]>0
    
        # Form the Y and X columns to be regressed
        Xdata = np.extract(mask,S[idx+1])
        Ydata = np.extract(mask,cFlows[idx+1])*np.extract(mask,disc[idx+1])
        
        if method_type =="LS":
            
             if len(Xdata) == 1:
                 coeffs = Ydata             
             elif len(Xdata) == 0:           
                 coeffs = []           
             else:  
                 # Do the regression, in this case using a quadratic fit
                 coeffs = np.polyfit(Xdata,Ydata,np.minimum(2,len(Xdata)-1))
            
             # Calculate potential payoff from "backwards induction"
             payoffs = np.polyval(coeffs,Xdata)
            
        elif method_type =="NN":
        
             clf = MLPRegressor(verbose=False,hidden_layer_sizes=(20,20,20,20),
                   warm_start=False, early_stopping=True, activation ='relu', 
                   shuffle =False, tol=0.000001,  max_iter = 15)
             
             if len(Xdata) == 1:         
                 payoffs = Ydata          
             elif len(Xdata) == 0:         
                 payoffs = []           
             else:               
                 clf.fit(Xdata.reshape(-1, 1),Ydata)
                 payoffs = clf.predict(Xdata.reshape(-1, 1))
            
        elif method_type =="DT":
    
             clf = DecisionTreeRegressor( max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                   min_weight_fraction_leaf=0.0, max_features=None, 
                   random_state=None, max_leaf_nodes=None, 
                   min_impurity_decrease=0.0, min_impurity_split=None, 
                   presort=False)
             
             if len(Xdata) == 1:          
                 payoffs = Ydata           
             elif len(Xdata) == 0:         
                 payoffs = []          
             else:
                 clf.fit(Xdata.reshape(-1, 1),Ydata)
                 payoffs = clf.predict(Xdata.reshape(-1, 1))
            
        elif method_type =="SVM":
             clf = svm.SVR()
             
             if len(Xdata) == 1:          
                 payoffs = Ydata           
             elif len(Xdata) == 0:          
                 payoffs = []           
             else:
                 clf.fit(Xdata.reshape(-1, 1),Ydata)
                 payoffs = clf.predict(Xdata.reshape(-1, 1))
            
        elif method_type =="SGD":
             clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
             
             if len(Xdata) == 1:         
                 payoffs = Ydata          
             elif len(Xdata) == 0:         
                 payoffs = []           
             else:             
                 clf.fit(Xdata.reshape(-1, 1),Ydata)
                 payoffs = clf.predict(Xdata.reshape(-1, 1))
            
        elif method_type =="ISO":
             clf = IsotonicRegression()
             
             if len(Xdata) == 1:      
                 payoffs = Ydata       
             elif len(Xdata) == 0:            
                 payoffs = []             
             else:                 
                 clf.fit(Xdata,Ydata)
                 payoffs = clf.predict(Xdata)
             
        elif method_type =="GPR":
             clf = GaussianProcessRegressor()
             
             if len(Xdata) == 1:           
                 payoffs = Ydata            
             elif len(Xdata) == 0:           
                 payoffs = []          
             else:          
                 clf.fit(Xdata.reshape(-1, 1),Ydata)
                 payoffs = clf.predict(Xdata.reshape(-1, 1))
            
        # Find location(s) where early exercise is optimal
        eeLoc = np.extract(mask,cFlows[idx,:]) > payoffs
        # Update the cash flow matrix to account for early exercise
        mask[mask] = eeLoc # These locations get exercised early
        cFlows[idx,:] = mask*cFlows[idx,:] # These are the cash flows
        cFlows[idx+1:,:] = cFlows[idx+1:,:]*np.logical_not(mask)
        
    # Discount each cash flow and average
    oPrice = np.matmul(np.cumprod(disc.mean(1)),np.transpose((np.sum(cFlows,axis=1)/nPaths)))
    
    return oPrice
