import data as dt
import pandas as pd 
import numpy as np 
from scipy.optimize import minimize

tickers = ["PE500.PA", "PANX.PA", "RS2K.PA", "DCAM.PA", "PTPXH.PA"]

def ew(tickers):
    """Portfolio équipondéré (Equal Weight)"""
    n = len(tickers)
    return pd.Series(np.repeat(1.0 / n, n), index=tickers, name="EW")


def min_variance_portfolio(tickers, short=False, leverage=True, max_weight=1.0):
    """
    Calcule le portefeuille à variance minimale (GMV).
    
    Parameters:
    -----------
    tickers : list
        Liste des tickers
    short : bool
        Autorise les positions courtes (short selling)
    leverage : bool
        Autorise le levier (si False, gross exposure ≤ 1)
    max_weight : float
        Poids maximum par actif (en valeur absolue si short=True)
    
    Returns:
    --------
    pd.Series : Poids optimaux du portefeuille
    """
    n = len(tickers)
    initial_weights = np.repeat(1/n, n)

    # Matrice de covariance
    cov_matrix = src.covariance_matrix(tickers).values
    
    # Vérification que la matrice est semi-définie positive
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < -1e-10):
        raise ValueError("La matrice de covariance n'est pas semi-définie positive")

    # Fonction objectif : variance du portefeuille
    def portfolio_var(weights):
        return float(weights.T @ cov_matrix @ weights)

    # Contrainte : somme des poids = 1 (fully invested)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Contrainte de levier si short autorisé mais pas de leverage
    if short and not leverage:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: 1.0 - np.sum(np.abs(w))  # gross exposure ≤ 1
        })

    # Définition des bornes
    if short:
        bounds = tuple([(-max_weight, max_weight) for _ in range(n)])
    else:
        bounds = tuple([(0, max_weight) for _ in range(n)])

    # Optimisation
    result = minimize(
        fun=portfolio_var,
        x0=initial_weights,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    # Vérification de la convergence
    if not result.success:
        error_msg = result.get('message', 'Optimisation failed')
        raise ValueError(f"L'optimisation a échoué: {error_msg}")

    # Nom descriptif selon les paramètres
    if short and leverage:
        name = "GMV_short_leveraged"
    elif short and not leverage:
        name = "GMV_short_no_leverage"
    else:
        name = "GMV_long_only"
    
    weights = pd.Series(result.x, index=tickers, name=name)
    
    return weights


def portfolio_return(tickers, weights):
    return weights @ src.annual_returns(tickers)

def portfolio_mu(tickers, weights):
    return weights @ src.mu(tickers)

def portfolio_vol(tickers, weights):
    return (weights.T @ src.covariance_matrix(tickers) @ weights)**0.5

def sharpe_ratio(tickers, weights): 
    return (portfolio_mu(tickers, weights) - src.get_risk_free()) / portfolio_vol(tickers, weights)

print(portfolio_vol(tickers, min_variance_portfolio(tickers)))

