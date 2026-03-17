#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install yfinance pandas numpy


# In[4]:


import yfinance as yf
import pandas as pd

# Stock list
stocks = ['AAPL','MSFT','GOOGL','JPM','XOM','JNJ','WMT','TSLA','T','BA','^GSPC']


# In[5]:


# Download data
data = yf.download(stocks, start='2018-01-01', end='2024-12-31')

# Show first few rows
data.head()


# In[7]:


# Save file 

data.to_csv('F:/zalima internship/project 2/raw_stock_data.csv')

print("✅ Dataset saved successfully!")


# In[9]:


print(data.columns)


# In[11]:


data.columns.levels


# In[13]:


data.columns.get_level_values(0).unique()


# In[15]:


#Extract close prices
adj_close = data['Close']
adj_close.head()


# In[16]:


# Fill missing values
adj_close = adj_close.fillna(method='ffill')

# Drop remaining nulls
adj_close = adj_close.dropna()


# In[17]:


adj_close.to_csv('F:/zalima internship/project 2/cleaned_stock_data.csv')

print("✅ Cleaned data saved successfully!")


# In[ ]:




