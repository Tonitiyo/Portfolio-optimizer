from src.estimators_models.expected_returns import historical_returns
from src.data.volatility import volatility

# Sharpe Ratio
def sharpe_ratio(tickers, rf ):
    er = historical_returns(tickers)
    vol = volatility(tickers)
    sharpe = (er - rf) / vol.iloc[-1]
    return sharpe