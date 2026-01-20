import pandas as pd
import source as src
import optimization_function as opt
import matplotlib.pyplot as plt

#Backtesting Version 1
def backtesting(tickers, model, recalibration="Quarterly"):
    returns = src.returns(tickers)
    frequency = {
        "Daily": 1,
        "Weekly": 5,
        "Monthly": 21,
        "Quarterly": 63,
        "Annualy":252
    }
    step = frequency[recalibration]
    portfolio_returns = pd.Series(index=returns.index)

    for i, date in enumerate(returns.index):
        if i % step == 0:
            weights = model(tickers)

            if not isinstance(weights, pd.Series):
                weights = pd.Series(weights, index=tickers)

        portfolio_returns.loc[date]= (returns.iloc[i] * weights).sum()

    net_asset_value = (1 + portfolio_returns).cumprod()

    return weights, net_asset_value

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]

    # GMV backtest
    w_gmv, nav_gmv = backtesting(tickers, opt.min_variance_portfolio, recalibration="Annualy")
    # Equal weight backtest
    w_eq, nav_eq = backtesting(tickers, opt.ew, recalibration="Annualy")

    print("GMV weights:")
    print(w_gmv)
    print("\nEqual weights:")
    print(w_eq)

    plt.figure(figsize=(10, 5))
    plt.plot(nav_gmv, label="GMV Portfolio")
    plt.plot(nav_eq, label="Equally Weighted Portfolio")
    plt.title("GMV vs Equally Weighted Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (starting at 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
