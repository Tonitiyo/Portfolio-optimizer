from src.data.returns import returns

# Matrice de corrélation
def correlation_matrix(tickers):
    returns_data = returns(tickers)
    corr_matrix = returns_data.corr()
    return corr_matrix

# Matrice de covariance
def covariance_matrix(tickers):
    returns_data = returns(tickers)
    cov_matrix = returns_data.cov()
    return cov_matrix