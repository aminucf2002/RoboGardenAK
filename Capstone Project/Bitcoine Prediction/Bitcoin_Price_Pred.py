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
# https://www.kaggle.com/code/richardgg93/rnn-example

#%% import data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Cryptocurrency Prices by Date.csv')
df = pd.read_csv(file_path)

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


#create df_m that has date column with monthly frequency and price column with mean price for that month for each currency
df_m = (
    df
    .set_index('Date')
    .groupby('Currency')['Price']
    .resample('ME')                # monthly frequency
    .mean()                      # mean price per month
    .reset_index()
)

monthly_ohlc = (
    df
    .set_index('Date')
    .groupby('Currency')['Price']
    .resample('ME')
    .agg(['first', 'last'])   # first = open, last = close
    .reset_index()
    .rename(columns={
        'first': 'open price',
        'last': 'close price'
    })
)
df_m = df_m.merge(
    monthly_ohlc,
    on=['Currency', 'Date'],
    how='left'
)
df_m['month_diff'] = df_m['close price'] - df_m['open price']


btc_df= df[df['Currency'] == 'bitcoin'].copy()
btc_df_m=df_m[df_m['Currency'] == 'bitcoin'].copy()


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

py.plot(go.Figure(data=data, layout=layout), filename='basic-line.html', auto_open=False)
#show the plot in notebook
py.iplot(go.Figure(data=data, layout=layout))
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

py.plot(go.Figure(data=data, layout=layout), filename='bitcoin-price.html', auto_open=False)


# https://www.kaggle.com/code/richardgg93/rnn-example




# %%
