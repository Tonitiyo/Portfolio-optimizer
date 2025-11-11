import source as src
import pandas as pd 
import numpy as np 
from scipy.optimize import minimize

tickers = ["PE500.PA", "PANX.PA", "RS2K.PA", "DCAM.PA", "PTPXH.PA"]

def ew(tickers):
    """Equally weighted portfolio """
    n = len(tickers)
    return pd.Series(np.repeat(1.0 / n, n), index=tickers, name="EW")


def min_variance_portfolio(tickers, short=False, leverage=True, max_weight=1.0):
    """
    Calculate weights for a min variance portfolio (GMV = Global Minimum Variance)

    Parameters:
    
    tickers : list
    short : bool
        Allows short position 
    leverage : bool
        Allows leverage  
    max_weight : float
        Max weight per assets 
    
    Returns:
    --------
    pd.Series : Optimal weights for our portfolio
    """

    n = len(tickers)
    initial_weights = np.repeat(1/n, n)

    #Covariance matrix
    cov_matrix = src.covariance_matrix(tickers).values
    
    #Varify if the semi defined matrix is positive
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < -1e-10):
        raise ValueError("La matrice de covariance n'est pas semi-définie positive")

    # Goal fonction - variance of the portfolio 
    def portfolio_var(weights):
        return float(weights.T @ cov_matrix @ weights)

    # Constrains : sum of weights =1 
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Leverage constrains if short allowed but no leverage 
    if short and not leverage:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: 1.0 - np.sum(np.abs(w))  # gross exposure ≤ 1
        })

    # Defines limits
    if short:
        bounds = tuple([(-max_weight, max_weight) for _ in range(n)])
    else:
        bounds = tuple([(0, max_weight) for _ in range(n)])

    # Optimization
    result = minimize(
        fun=portfolio_var,
        x0=initial_weights,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    # Verify convergence
    if not result.success:
        error_msg = result.get('message', 'Optimisation failed')
        raise ValueError(f"L'optimisation a échoué: {error_msg}")

    # Desciption for each parameter
    if short and leverage:
        name = "GMV_short_leveraged"
    elif short and not leverage:
        name = "GMV_short_no_leverage"
    else:
        name = "GMV_long_only"
    
    weights = pd.Series(result.x, index=tickers, name=name)
    
    return weights

# Porfolio calculation for comparison 
def portfolio_return(tickers, weights):
    return weights @ src.annual_returns(tickers)

def portfolio_mu(tickers, weights):
    return weights @ src.mu(tickers)

def portfolio_vol(tickers, weights):
    return (weights.T @ src.covariance_matrix(tickers) @ weights)**0.5

def sharpe_ratio(tickers, weights): 
    return (portfolio_mu(tickers, weights) - src.get_risk_free()) / portfolio_vol(tickers, weights)


