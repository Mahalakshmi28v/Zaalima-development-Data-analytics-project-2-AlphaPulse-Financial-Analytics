#!/usr/bin/env python
# coding: utf-8

# In[5]:


import yfinance as yf
import pandas as pd

# Stock list
stocks = ['AAPL','MSFT','GOOGL','JPM','XOM','JNJ','WMT','TSLA','T','BA','^GSPC']

# Download latest 1 year data
data = yf.download(stocks, period='1y')

# Extract Close prices
close_data = data['Close']

# Save file
close_data.to_csv(r'F:\zalima internship\project 2\latest_stock_data.csv')

print("✅ Data refreshed successfully!")


# In[4]:


import pandas as pd
import numpy as np

# Load data (USE FULL PATH)
data = pd.read_csv(r'F:\zalima internship\project 2\cleaned_stock_data.csv', index_col=0)

# Calculate returns
returns = data.pct_change().dropna()

# Equal weights
weights = np.ones(len(returns.columns)) / len(returns.columns)

# Portfolio returns
portfolio_returns = returns.dot(weights)

# VaR
var_95 = np.percentile(portfolio_returns, 5)

# Volatility
volatility = portfolio_returns.std()

# Return
mean_return = portfolio_returns.mean()

# Max Drawdown
cum_returns = (1 + portfolio_returns).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("Return:", mean_return)
print("Volatility:", volatility)
print("VaR (95%):", var_95)
print("Max Drawdown:", max_drawdown)


# In[6]:


final_metrics = pd.DataFrame({
    'Metric': ['Annual Return', 'Volatility', 'VaR (95%)', 'Max Drawdown'],
    'Value': [mean_return, volatility, var_95, max_drawdown]
})

final_metrics.to_csv(r'F:\zalima internship\project 2\portfolio_metrics_final.csv', index=False)


# In[ ]:




