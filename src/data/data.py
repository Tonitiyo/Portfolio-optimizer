import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as date
import scipy.stats as stats

def get_prices(tickers, start="2000-01-01", end=None):
    if end is None:
        end=date.datetime.today()
    
    prices = yf.download(tickers, start=start, end=end)["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name="Close")
    
    prices = prices.dropna()
    return prices

def get_risk_free(start="2000-01-01", end=None):
    if end is None:
        end = date.datetime.today()

    rf_3m = yf.download("^IRX", start=start, end=end)["Close"] / 100
    rf_10y = yf.download("^TNX", start=start, end=end)["Close"] / 100

    risk_free = pd.Series({"RiskFree_3M": rf_3m, "RiskFree_10Y": rf_10y})
    risk_free = risk_free.dropna()
    return risk_free


# Matrice de corrélation
def correlation_matrix(tickers):
    returns_data = returns(tickers)
    corr_matrix = returns_data.corr()
    return corr_matrix

# Matrice de covariance
def covariance_matrix(tickers):
    returns_data = returns(tickers)
    cov_matrix = returns_data.cov()
    return cov_matrix

# Annualized expected returns
def mu(tickers):
    returns_data = returns(tickers)
    mean_returns = returns_data.mean() * 252  # Annualized mean return
    return mean_returns

# Daily expected returns
def day_mu(tickers):
    returns_data = returns(tickers)
    mean_returns = returns_data.mean()
    return mean_returns

# Sharpe Ratio

def sharpe_ratio(tickers, rf ):
    er = mu(tickers)
    vol = volatility(tickers)
    sharpe = (er - rf) / vol.iloc[-1]
    return sharpe

