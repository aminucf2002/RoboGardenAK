#%% importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlib inline

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
#from wordcloud import WordCloud
from collections import Counter
# from nltk.corpus import stopwords
# from nltk.util import ngrams
# stop = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler



import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

import os

#%% import data
df = pd.read_csv('Cryptocurrency Prices by Date.csv')

#convert Unix time ine #Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], unit='ms')

df['Date'] = df['Date'].dt.date # Extract date only (remove time if present)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

#find duplicated date for each currency and replace the price with mean price for that date
df = df.groupby(['Currency', 'Date']).agg({'Price': 'mean'}).reset_index()

#count the number of duplicated date for each currency
duplicated_counts = df.groupby(['Currency', 'Date']).size()
duplicated_counts = duplicated_counts[duplicated_counts > 1]
print("Number of duplicated date for each currency:")
print(duplicated_counts)
# Let's take a look at the data

# Create Year-Month key
df['YearMonth'] = df['Date'].dt.to_period('M')

# Compute monthly mean and SD per currency
df['month_mean'] = df.groupby(['Currency', 'YearMonth'])['Price'].transform('mean')
df['month_SD'] = df.groupby(['Currency', 'YearMonth'])['Price'].transform('std')
df['month_diff'] = df.groupby(['Currency', 'YearMonth'])['Price'].transform(lambda x: x.iloc[-1] - x.iloc[0])

btc_df= df[df['Currency'] == 'bitcoin'].copy()
# bt_df['month_diff'] = bt_df.groupby('YearMonth')['Price'].transform(lambda x: x.iloc[-1] - x.iloc[0])
# %% -------------------- At first let's take 10 random currencies with minimum Price > threshold and plot them.

np.random.seed(40)  # For reproducibility
threshold=10
eligible_currencies = df.loc[df['Price'] < 1000000, 'Currency'].unique()
selected_currencies = np.random.choice(eligible_currencies, min(30, len(eligible_currencies)), replace=False)


data = []
for currency in selected_currencies:
    currency_df = df[df['Currency'] == currency]
    if not currency_df.empty:
        data.append(go.Scatter(
            x=currency_df['Date'],
            y=currency_df['Price'],
            mode='lines',
            name=currency
        ))

layout = go.Layout(
    title=f"Price of {len(selected_currencies)} random assets",
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price (USD)'),
    legend=dict(orientation="h")
)

py.iplot(go.Figure(data=data, layout=layout), filename='basic-line')
#Currencies are sampled randomly, but you should see that some currencies started trading later 


# %% -------------------- Take bitcoin currencies  plot it.
data=[]
bitcoin_df=df[df['Currency']=='bitcoin']
data.append(go.Scatter(
            x=bitcoin_df['Date'],
            y=bitcoin_df['Price'],
            mode='lines',
            name="bitcoin"
        ))

layout = go.Layout(
    title="Bitcoin Price",
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price (USD)'),
    legend=dict(orientation="h")
)

py.iplot(go.Figure(data=data, layout=layout), filename='basic-line')


https://www.kaggle.com/code/richardgg93/rnn-example




# %%
