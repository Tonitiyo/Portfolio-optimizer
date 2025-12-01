import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display

# Parameters
S_0 = 40
r = .03
vol = .2
T = 1/3

# Calculate a call and put option using Monte Carlo simulation
def monte_carlo_call_put(S_0, r, vol, T, K, n_simulations=10000):
    # Simulate end-of-period stock prices
    Z = np.random.normal(0, 1, n_simulations)
    S_T = S_0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)

    # Calculate call and put payoffs
    call_payoffs = np.maximum(S_T - K, 0)
    put_payoffs = np.maximum(K - S_T, 0)
    
    # Discount payoffs back to present value
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_price = np.exp(-r * T) * np.mean(put_payoffs)
    
    return call_price, put_price
"""
print(monte_carlo_call_put(S_0, r, vol, T, K=40))
print(monte_carlo_call_put(S_0, r, vol, T, K=40))
"""


#Pricing options using B&S formula
def BS_call_put(S_0, r, vol, T, K):
    #Calculate d1 and d2
    d1 = (np.log(S_0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    call_price = S_0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S_0 * stats.norm.cdf(-d1)
    
    return call_price, put_price

def results(S_0, r, vol, T, K):
    mc_results = monte_carlo_call_put(S_0, r, vol, T, K)
    bs_results = BS_call_put(S_0, r, vol, T, K)

    results_df = pd.DataFrame({
        'Methods': ['Monte Carlo', 'Black-Scholes'],
        'Call Price': [mc_results[0], bs_results[0]],
        'Put Price': [mc_results[1], bs_results[1]]
    })
    return results_df

print(results(S_0, r, vol, T, K = 40))

# export results to csv file
results_df = results(S_0, r, vol, T, K = 40)
results_df.to_csv('option_pricing_results.csv', index=False)
