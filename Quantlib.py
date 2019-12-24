#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:18:03 2019

@author: Goutham Balaraman
Website: http://gouthamanbalaraman.com/blog/american-option-pricing-quantlib-python.html
"""
import QuantLib as ql 

def quantlibprice(strike_price,optiontype,volatility):

    # option data
    maturity_date = ql.Date(15, 11, 2020)
    spot_price = 311.790009
    dividend_rate =  0
    
    if optiontype == "call":
    
        option_type = ql.Option.Call
        
    elif optiontype == "put":
    
        option_type = ql.Option.Put
    
    risk_free_rate = 0.0154
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()
    
    calculation_date = ql.Date(15, 11, 2019)
    ql.Settings.instance().evaluationDate = calculation_date
    
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    settlement = calculation_date
    
    am_exercise = ql.AmericanExercise(settlement, maturity_date)
    american_option = ql.VanillaOption(payoff, am_exercise)
    
    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, volatility, day_count)
    )
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                               dividend_yield, 
                                               flat_ts, 
                                               flat_vol_ts)
    
    steps = 200
    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    american_option.setPricingEngine(binomial_engine)
    return american_option.NPV()