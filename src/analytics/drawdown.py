import pandas as pd

def drawdown(return_series: pd.Series):
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    
    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns
                         })

if __name__ == "__main__":
    # super simple test (5 points)
    returns = pd.Series([0.10, -0.20, 0.05, 0.00, 0.10])
    print(drawdown(returns).round(4))