'''
This script acts as a calculator for the Black Scholes Implied Volatility, using the Newton Raphson root finding method.
'''

from scipy.special import ndtr
from scipy.stats import norm
from scipy.optimize import newton
import numpy as np

N = ndtr
N_prime = norm._pdf

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * N(d1) - np.exp(-r * T) * K * N(d2)

def bs_vega(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return S * N_prime(d1) * np.sqrt(T)

def find_vol(price, S, K, T, r, *args):
    ''''
    Finds Black Scholes implied volatility of a European call option using Newton Raphson root finding method.

    price: float
    Value of the option

    S: float
    Spot price of the asset

    K: float
    Strike price of the option

    T: float
    Time to maturity in years

    r: float
    Risk-free interest rate
    '''

    MAX_STEPS = 200
    PRECISION = 1.0e-5 
    sigma = 0.5 # first guess of implied volatility
    for i in range(0, MAX_STEPS):
        bs_price = bs_call(S, K, T, r, sigma) # bs price
        vega = bs_vega(S, K, T, r, sigma)
        diff = price - bs_price
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff / vega
    return sigma




