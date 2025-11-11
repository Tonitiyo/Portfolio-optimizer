from data import get_prices 
import numpy as np


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

# Quick test

if __name__ == "__main__":
    print("Running quick test…", flush=True)  # prints immediately
    tickers = ["AAPL", "MSFT", "TSLA"]

    vol = volatility(tickers)
    print("\nVolatility (tail):")
    print(vol.tail(), flush=True)

    daily_vols = daily_vol(tickers)
    print("\nDaily Volatility (tail):")
    print(daily_vols.tail(), flush=True)

