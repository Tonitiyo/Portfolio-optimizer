from src.data import returns
from src.data import get_prices

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

# Quick test
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]
    print(get_prices(tickers))
    print(kurtosis(tickers))
    print(skewness(tickers))


