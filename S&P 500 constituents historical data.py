### S&P 500 constituents historical data 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as date
import scipy.stats as stats
import requests # To fetch the Wikipedia page

# Retrieve ticker symbols for S&P 500 constituents from Wikipedia
def get_stock_data():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ### instal pip install lxml if you don't have it
    # Wikipedia blocks pandas.read_html by default, so we use requests to get the page content
    # We are using a user-agent to mimic a browser visit
    headers = {'User-Agent': 'Mozilla/5.0'}
    #Install requests if you don't have it and import it
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text)
    sp500_tickers = tables[0]["Symbol"].tolist()
    # Some tickers have dots in them which yfinance does not recognize, replace them with hyphens
    data_constituents = yf.download(sp500_tickers, start='2000-01-01', end=date.datetime.today(), group_by='ticker')
    data_constituents = data_constituents.dropna(how='all')  # Drop rows where all elements are NaN
    return data_constituents

SP500_constituents = get_stock_data()
print(SP500_constituents["MSFT"].loc["2025"].head())

# Calculate returns 
def calculate_returns(data):
    returns = pd.DataFrame()
    for ticker in data.columns.levels[0]:
        returns[ticker] = data[ticker]['Close'].pct_change()
    returns = returns.dropna()
    return returns

# 
# Calculate covariance matrix
def calculate_covariance(returns):
    covariance_matrix = returns.cov() * 252  # Annualize the covariance matrix
    return covariance_matrix
