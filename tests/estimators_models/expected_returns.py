### Optimization function 
from src.data.returns import daily_returns

# Annualized expected returns
def historical_returns(tickers):
    return daily_returns(tickers).mean() * 252

# Daily expected returns
def day_mu(tickers):
    returns_data = daily_returns(tickers)
    mean_returns = returns_data.mean()
    return mean_returns