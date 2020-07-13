import pandas as pd
import numpy as np

df=pd.read_csv('AAPL.csv')
df.head()
df.shape()
#We only require the open and date column so we drop all others
df.drop(['High','Low','Close','Adj'],axis=1)
df['Year'] = df['Volume'].apply(to_int)
df['Year'] = df['Open'].apply(to_int)
df['Date'] = df['Date'].apply(to_str)
#We regularize the dataset by a volume-price scale
df.fillna(df.mean(),axis=1)
df['prices']=df['Open']/df['Volume']
#Removal of outliers
df = df[(np.abs(stats.zscore(df.drop(['Date'], axis=1))) < 3).all(axis=1)]
#String to date-Time format
df['DATE'] = df['DATE'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))



