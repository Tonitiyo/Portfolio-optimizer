import pandas as pd 
import yfinance as yf
import datetime as date
import numpy as np 

"""
0. Get data from yahoo finance (stocks + risk free)
"""


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



"""
1. Return functions
"""
def returns(ticker):
    returns = get_prices(ticker).pct_change().dropna()
    return returns 

def annual_returns(tickers):
    returns_data = returns(tickers)
    annual_rets = returns_data.mean() * 252
    return annual_rets

"""
2. Volatility functions
"""

def volatility(tickers, window=21):
    prices = get_prices(tickers)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    return vol

def daily_vol(tickers, window=21):
    prices = get_prices(tickers)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_vol = log_returns.rolling(window=window).std()
    return daily_vol

"""
3. Useful function for portfolio optimizer
"""

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

