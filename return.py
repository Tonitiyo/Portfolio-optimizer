from data import get_prices 

def returns(ticker):
    returns = get_prices(ticker).pct_change().dropna()
    return returns 

def annual_returns(tickers):
    returns_data = returns(tickers)
    annual_rets = returns_data.mean() * 252
    return annual_rets

# Quick test
# print(returns("AAPL"), annual_returns("AAPL"))


if __name__ == "__main__":
    print("Running quick test…", flush=True)  # prints immediately
    tickers = ["AAPL", "MSFT", "TSLA"]

    rets = returns(tickers)
    print("\nReturns (tail):")
    print(rets.tail(), flush=True)

    ann_rets = annual_returns(tickers)
    print("\nAnnual Returns:")
    print(ann_rets, flush=True)
    

    