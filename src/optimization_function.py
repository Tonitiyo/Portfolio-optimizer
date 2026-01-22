import data as dt
import pandas as pd 
import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt

tickers = ["PE500.PA", "PANX.PA", "RS2K.PA", "DCAM.PA", "PTPXH.PA"]

#Equally weighted

def ew(tickers):
    """Equally weighted portfolio """
    n = len(tickers)
    return pd.Series(np.repeat(1.0 / n, n), index=tickers, name="EW")

#Define constraints and borns 

def bld_constraints(n, short=False, leverage=False):
     # Constrains : sum of weights =1 
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Leverage constrains if short allowed but no leverage 
    if short and not leverage:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: 1.0 - np.sum(np.abs(w))  # gross exposure ≤ 1
        })
    return constraints

def bld_bounds(n, short= False, max_weight=1.0):
    if short:
        bounds = tuple([(-max_weight, max_weight) for _ in range(n)])
    else:
        bounds = tuple([(0, max_weight) for _ in range(n)])
    return bounds

#GMV portfolio

def min_variance_portfolio(tickers, short=False, leverage=True, max_weight=1.0, tol=1e-8):
    """
    Calculate weights for a min variance portfolio (GMV = Global Minimum Variance)
    """

    n = len(tickers)
    initial_weights = np.repeat(1/n, n)

    #Covariance matrix
    cov_matrix = dt.covariance_matrix(tickers).values
    
    #Varify if the semi defined matrix is positive
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < -1e-10):
        raise ValueError("La matrice de covariance n'est pas semi-définie positive")

    # Goal fonction - variance of the portfolio 
    def portfolio_var(weights):
        return float(weights.T @ cov_matrix @ weights)

    constraints = bld_constraints(n, short=short, leverage=leverage)
    bounds = bld_bounds(n, short=short, max_weight=max_weight)

    # Optimization
    result = minimize(
        fun=portfolio_var,
        x0=initial_weights,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    w = result.x
    w = (np.where(np.abs(w) < tol, 0, w)); w = w / w.sum()
    return pd.Series(w, index=tickers, name="Global minimum variance")

#Risk parity

def risk_parity(tickers, short=False, leverage=False, max_weight=1, tol = 1e-8): 
    
    n = len(tickers)
    initial_weights = np.repeat(1/n, n)
    
    cov_matrix = dt.covariance_matrix(tickers)
    
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < -1e-10):
        raise ValueError("La matrice de covariance n'est pas semi-définie positive")
    
    def objective(weights, cov_matrix):
        port_var = weights.T @ cov_matrix @ weights
        port_vol = np.sqrt(port_var)
        
        # Avoid to divide by 0
        if port_vol < 1e-10:
            return 1e10
        
        # marginal contributions
        mrc = cov_matrix @ weights 
        
        # risk contributions
        rc = (weights * mrc)/port_vol
        
        # target
        target_rc = port_vol / n
        
        return np.sum((rc - target_rc)**2)
    
    constraints = bld_constraints(n, short=short, leverage=leverage)
    bounds = bld_bounds(n, short=short, max_weight=max_weight)
    
    # Optimization
    result = minimize(
        fun=objective,
        x0=initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if not result.success:
        error_msg = result.get('message', 'Optimisation failed')
        raise ValueError(f"L'optimisation a échoué: {error_msg}")
    
    w = result.x
    w = (np.where(np.abs(w) < tol, 0, w)); w = w / w.sum()

    return pd.Series(w, index=tickers, name="Risk parity")

#MSR 

def msr(tickers, short=False, leverage=False, max_weight=1, tol = 1e-8):
    n = len(tickers)
    initial_weights = np.repeat(1/n, n)

    # Calculate parameters 
    cov_matrix = dt.covariance_matrix(tickers)
    mu = dt.mu(tickers)
    rf = dt.get_risk_free().iloc[-1, 0]

    def objective(weights):  
        port_rets = weights @ mu
        port_var = weights.T @ cov_matrix @ weights
        port_vol = np.sqrt(port_var)
        
        if port_vol < 1e-10:
            return 1e10
        
        sharpe = (port_rets - rf) / port_vol
        return -sharpe  # negative as we use the min function

    constraints = bld_constraints(n, short=short, leverage=leverage)
    bounds = bld_bounds(n, short=short, max_weight=max_weight)

    result = minimize(
        fun=objective,
        x0=initial_weights,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if not result.success:
        raise ValueError(f"L'optimisation a échoué: {result.message}")

    w = result.x
    w = (np.where(np.abs(w) < tol, 0, w)); w = w / w.sum()

    return pd.Series(w, index=tickers, name="Max Sharpe Ratio")

"""print(msr(tickers))"""

#Max decorrelation

def max_decorr(tickers, short=False, leverage=False, max_weight=1, tol = 1e-8):

    n = len(tickers)
    initial_weights = np.repeat(1/n, n)
    corr_matrix = dt.correlation_matrix(tickers)

    def objective(weights, corr_matrix):
        return weights.T @ corr_matrix @ weights
    
    constraints = bld_constraints(n, short=short, leverage=leverage)
    bounds = bld_bounds(n, short=short, max_weight=max_weight)

    result = minimize(
        fun = objective, 
        x0 = initial_weights,
        args = (corr_matrix,),
        method= 'SLSQP',
        constraints=constraints,
        bounds= bounds,
        options = {'ftol': 1e-9, 'maxiter': 1000}
    )

    w = result.x
    w = (np.where(np.abs(w) < tol, 0, w)); w = w / w.sum()

    return pd.Series(w, index=tickers, name="Max decorrelation")


#Max diversification 

def max_div(tickers, short=False, leverage=False, max_weight=1, tol = 1e-8):

    n = len(tickers)
    initial_weights = np.repeat(1/n, n)
    cov_matrix = dt.covariance_matrix(tickers)
    vols = np.sqrt(np.diag(cov_matrix))

    def objective(weights, cov_matrix):
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        num = weights @ vols
        if port_vol < 1e-12:
            return 1e12
        return - num / port_vol

    constraints = bld_constraints(n, short=short, leverage=leverage)
    bounds = bld_bounds(n, short=short, max_weight=max_weight)

    result = minimize(
        fun = objective, 
        x0 = initial_weights,
        args = (cov_matrix,),
        method= 'SLSQP',
        constraints=constraints,
        bounds= bounds,
        options = {'ftol': 1e-9, 'maxiter': 1000}
    )

    w = result.x
    w = (np.where(np.abs(w) < tol, 0, w)); w = w / w.sum()

    return pd.Series(w, index=tickers, name="Max diversification")

#Inverse volatility 

def inverse_vol(tickers, short=False, leverage=False, max_weight=1, tol=1e-8):
    cov_matrix = dt.covariance_matrix(tickers)
    vols = np.sqrt(np.diag(cov_matrix)) 
    inv_vols = 1/ vols
    weights = inv_vols / inv_vols.sum()

    return pd.Series(weights,
                     index=tickers,
                     name="Inverse Volatility")


#Result and comparison

def port_opti_result(tickers, short=False, leverage=False, max_weight=1, tol=1e-8):
    ew_w   = ew(tickers)
    gmv_w  = min_variance_portfolio(tickers, short=short, leverage=leverage, max_weight=max_weight, tol=tol)
    rp_w   = risk_parity(tickers, short=short, leverage=leverage, max_weight=max_weight, tol=tol)
    msr_w  = msr(tickers, short=short, leverage=leverage, max_weight=max_weight, tol=tol)
    mdcr_w = max_decorr(tickers, short=short, leverage=leverage, max_weight=max_weight, tol=tol)
    mdv_w  = max_div(tickers, short=short, leverage=leverage, max_weight=max_weight, tol=tol)
    inv_w  = inverse_vol(tickers, short=short, leverage=leverage, max_weight=max_weight, tol=tol)
    weights = pd.concat([ew_w, gmv_w, rp_w, msr_w, mdcr_w, mdv_w, inv_w], axis=1).T
    return weights.round(4)

print(port_opti_result(tickers))

"""port_opti_result(tickers).T.plot(kind="bar", figsize=(10,5))
plt.title("Best allocation per strategy")
plt.ylabel("Weights")
plt.grid(True)
plt.show()"""

#Black Litterman
