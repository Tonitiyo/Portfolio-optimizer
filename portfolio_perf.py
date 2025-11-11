#Portfolio performance
import numpy as np
import source as src
import optimization_function as optf

tickers = ["PE500.PA", "PANX.PA", "RS2K.PA", "DCAM.PA", "PTPXH.PA"]

def portfolio_return(tickers, weights):
    return weights @ src.annual_returns(tickers)

def portfolio_mu(tickers, weights):
    return weights @ src.mu(tickers)

def portfolio_vol(tickers, weights):
    return ((weights.T @ src.covariance_matrix(tickers) @ weights)**0.5)*np.sqrt(252)

def sharpe_ratio(tickers, weights): 
    return (portfolio_mu(tickers, weights) - src.get_risk_free().iloc[-1,0]) / portfolio_vol(tickers, weights)

#Exemple with the mean variance portfolio
"""
print(sharpe_ratio(tickers, optf.min_variance_portfolio(tickers)))
"""