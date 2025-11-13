import numpy as np
from scipy.stats import norm
from src.data.returns import daily_returns

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

# Quick test
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]
    print(historical_VaR(daily_returns(tickers)))
    print(VaR(daily_returns(tickers)))


