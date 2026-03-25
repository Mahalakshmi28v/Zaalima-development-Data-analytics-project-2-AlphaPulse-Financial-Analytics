#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install yfinance pandas numpy


# In[5]:


import yfinance as yf
import pandas as pd

# Stock list
stocks = ['AAPL','MSFT','GOOGL','JPM','XOM','JNJ','WMT','TSLA','T','BA','^GSPC']


# In[6]:


# Download data
data = yf.download(stocks, start='2018-01-01', end='2024-12-31')

# Show first few rows
data.head()


# In[7]:


# Save file 

data.to_csv('F:/zalima internship/project 2/raw_stock_data.csv')

print("✅ Dataset saved successfully!")


# In[8]:


print(data.columns)


# In[9]:


data.columns.levels


# In[10]:


data.columns.get_level_values(0).unique()


# In[11]:


#Extract close prices
adj_close = data['Close']
adj_close.head()


# In[12]:


# Fill missing values
adj_close = adj_close.fillna(method='ffill')

# Drop remaining nulls
adj_close = adj_close.dropna()


# In[13]:


adj_close.to_csv('F:/zalima internship/project 2/cleaned_stock_data.csv')

print("✅ Cleaned data saved successfully!")


# In[14]:


#WEEK 2
adj_close.head()


# In[15]:


import numpy as np

log_returns = np.log(adj_close / adj_close.shift(1))
log_returns = log_returns.dropna()

log_returns.head()


# In[16]:


log_returns.to_csv('F:/zalima internship/project 2/log_returns.csv')

print("✅ Log returns saved!")


# In[17]:


# Number of stocks
num_stocks = adj_close.shape[1]

# Equal weights
weights = np.array([1/num_stocks] * num_stocks)

print(weights)


# In[18]:


# Average daily return
mean_returns = log_returns.mean()

# Expected portfolio return
portfolio_return = np.sum(mean_returns * weights)

print("Expected Daily Return:", portfolio_return)


# In[19]:


annual_return = portfolio_return * 252  # 252 trading days

print("Expected Annual Return:", annual_return)


# In[20]:


cov_matrix = log_returns.cov()

cov_matrix.head()


# In[21]:


portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

print("Portfolio Variance:", portfolio_variance)


# In[22]:


portfolio_volatility = np.sqrt(portfolio_variance)

print("Daily Volatility:", portfolio_volatility)


# In[23]:


annual_volatility = portfolio_volatility * np.sqrt(252)

print("Annual Volatility:", annual_volatility)


# In[24]:


num_simulations = 10000
num_days = 252  # 1 year

# Store results
simulation_results = np.zeros((num_days, num_simulations))


# In[25]:


for i in range(num_simulations):
    
    # Random daily returns
    simulated_returns = np.random.normal(
        loc=portfolio_return,
        scale=portfolio_volatility,
        size=num_days
    )
    
    # Convert to price path (start = 100)
    price_path = 100 * np.exp(np.cumsum(simulated_returns))
    
    simulation_results[:, i] = price_path


# In[26]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(simulation_results)
plt.title("Monte Carlo Simulation (Portfolio Value)")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.show()


# In[27]:


# Final values after 1 year
final_values = simulation_results[-1]

# 5% worst-case
VaR_95 = np.percentile(final_values, 5)

print("Value at Risk (95% confidence):", VaR_95)


# In[28]:


import pandas as pd

simulation_df = pd.DataFrame(simulation_results)
simulation_df.to_csv('F:/zalima internship/project 2/monte_carlo_simulation.csv', index=False)

print("✅ Simulation data saved!")


# In[29]:


summary = pd.DataFrame({
    "Metric": ["Expected Annual Return", "Annual Volatility", "VaR (95%)"],
    "Value": [annual_return, annual_volatility, VaR_95]
})

summary.to_csv('F:/zalima internship/project 2/portfolio_summary.csv', index=False)

print("✅ Summary saved!")


# In[30]:


#monte file small, as older contain 10000 simulations
simulation_df_small = simulation_df.iloc[:, :200]  # keep 200 simulations

simulation_df_small.to_csv(
    'F:/zalima internship/project 2/monte_carlo_simulation_small.csv',
    index=False
)

print("✅ Smaller file saved!")


# In[31]:


#week3
correlation_matrix = log_returns.corr()

correlation_matrix.head()


# In[32]:


correlation_matrix.to_csv('F:/zalima internship/project 2/correlation_matrix.csv')

print("✅ Correlation matrix saved!")


# In[33]:


rolling_volatility = log_returns.rolling(window=30).std()

rolling_volatility = rolling_volatility.dropna()

rolling_volatility.head()


# In[34]:


rolling_volatility.to_csv('F:/zalima internship/project 2/rolling_volatility.csv')

print("✅ Rolling volatility saved!")


# In[ ]:




