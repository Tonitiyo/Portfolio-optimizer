import pandas as pd
import numpy as np
from scipy.stats import norm

from data import returns

def kurtosis(tickers):
    r = returns(tickers)
    if r is None or r.empty:
        raise RuntimeError("No returns data fetched. Check data source / internet / ticker symbols.")
    dm = r - r.mean()
    sigma = r.std(ddof=0)
    exp4 = (dm**4).mean()
    return exp4 / (sigma**4) 

def skewness(tickers):
    r = returns(tickers)
    if r is None or r.empty:
        raise RuntimeError("No returns data fetched. Check data source / internet / ticker symbols.")
    dm = r - r.mean()
    sigma = r.std(ddof=0)
    exp4 = (dm**3).mean()
    return exp4 / (sigma**3)

def drawdown(return_series: pd.Series):
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    
    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns
                         })

#Historical Value at Risk
def historical_VaR(input, confidence_level=0.95):
    return np.percentile(input, (1-confidence_level)*100, axis=0) 

def historical_CVaR(input, confidence_level=0.95, ):

    return

#Variance-covariance methods
def VaR(input, confidence_level=0.95):
    mean_returns = np.mean(input, axis=0)
    std_dev = np.std(input, axis=0)
    z_score = norm.ppf(1-confidence_level)
    return mean_returns+z_score*std_dev