'''
This module provides functions to calculate the Black Scholes price, implied volatility and Greeks for a European option. 

Most are implemented according to the formulas shown in https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks. 

Greeks are defined as they are on Deribit: https://insights.deribit.com/education/introduction-to-option-greeks/.
'''

from scipy.special import ndtr
from scipy.stats import norm
from scipy.optimize import newton
import numpy as np

N = ndtr
N_prime = norm._pdf

def bs_price(flag, S, K, T, r, vol):
    '''
    Function that calculates the European call or put option price.

    Parameters
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)
    vol: float
        Black Scholes implied volatility as a fraction

    Returns
    ----------
    price: float
        Black Scholes price of the option
    '''
        
    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * N(d1) - np.exp(-r * T) * K * N(d2)

    if flag == 'c':
        return C
    elif flag == 'p':
        return C + K * np.exp(-r * T) - S

def bs_delta(flag, S, K, T, r, vol):
    '''
    Function that calculates the delta of an European call or put option using the closed form formula.

    Parameters
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)
    vol: float
        Black Scholes implied volatility as a fraction 

    Returns
    ----------
    delta: float
        Black Scholes delta of the option
    '''

    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    if flag == 'c':
        return N(d1)
    elif flag =='p':
        return N(d1) - 1

def bs_gamma(flag, S, K, T, r, vol):
    '''
    Function that calculates the gamma of an European call or put option.

    Parameters
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)
    vol: float
        Black Scholes implied volatility as a fraction 

    Returns
    ----------
    gamma: float
        Black Scholes gamma of the option
    '''

    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return (N_prime(d1)) / (S * vol * np.sqrt(T))

def bs_vega(flag, S, K, T, r, vol):
    '''
    Function that calculates the vega of an European call or put option, defined as the change in option price for a 1% change in implied volatility.

    Parameters
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)
    vol: float
        Black Scholes implied volatility as a fraction

    Returns
    ----------
    vega: float
        Black Scholes vega of the option
    '''

    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return S * N_prime(d1) * np.sqrt(T) / 100

def bs_theta(flag, S, K, T, r, vol):
    '''
    Function that calculates the theta of an European call or put option.

    Parameters
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)
    vol: float
        Black Scholes implied volatility as a fraction 

    Returns
    ----------
    theta: float
        Black Scholes theta of the option
    '''

    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if flag == 'c':
        return ( -(S * N_prime(d1) * vol) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2) ) / 365
    elif flag == 'p':
        return ( -(S * N_prime(d1) * vol) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N(-d2) ) / 365

def bs_rho(flag, S, K, T, r, vol):
    '''
    Function that calculates the rho of an European call or put option.

    Parameters
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)
    vol: float
        Black Scholes implied volatility as a fraction 

    Returns
    ----------
    rho: float
    Black Scholes rho of the option
    '''

    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if flag == 'c':
        return K * T * np.exp(-r * T) * N(d2) / 100
    elif flag == 'p':
        return -K * T * np.exp(-r * T) * N(-d2) / 100


def bs_impliedvol(flag, price, S, K, T, r, *args):
    ''''
    Finds Black Scholes implied volatility of a European call or put option using Newton Raphson root finding method.
    
    Attributes
    ----------
    flag: str
        'c' or 'p' for call or put option respectively
    price: float
        Price of the option
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    T: float
        Time to maturity (in years)
    r: float
        Risk-free interest rate as a fraction (e.g. 0.01 for 1%)

    Returns
    ----------
    impliedvol: float
    Black Scholes implied volatility of the option
    '''

    MAX_STEPS = 200
    PRECISION = 1.0e-5 
    sigma = 0.5 # first guess of implied volatility
    for i in range(0, MAX_STEPS):
        P = bs_price(flag, S, K, T, r, sigma) # bs price
        vega = bs_vega(flag, S, K, T, r, sigma) * 100
        diff = price - P
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff / vega
    return sigma

