#Portfolio performance
import numpy as np
import data as dt
import optimization_function as optf

tickers = ["PE500.PA", "PANX.PA", "RS2K.PA", "DCAM.PA", "PTPXH.PA"]

def portfolio_returns(tickers, weights):
    rets = dt.returns(tickers)
    weights = np.asarray(weights, dtype=float)
    return rets.dot(weights)

def portfolio_mu(tickers, weights):
    return weights @ dt.mu(tickers)

def portfolio_vol(tickers, weights):
    return ((weights.T @ dt.covariance_matrix(tickers) @ weights)**0.5)*np.sqrt(252)

def sharpe_ratio(tickers, weights): 
    return (portfolio_mu(tickers, weights) - dt.get_risk_free().iloc[-1,0]) / portfolio_vol(tickers, weights)

#Exemple with the mean variance portfolio

"""print(portfolio_returns(tickers, optf.min_variance_portfolio(tickers)).iloc["2001:"])"""
print(portfolio_mu(tickers, optf.min_variance_portfolio(tickers)))
print(dt.returns(tickers).loc["2001:"])