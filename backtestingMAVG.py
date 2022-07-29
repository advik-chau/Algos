#%%
import pandas as pd
import pandas_datareader as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import backtrader as bt

#%%
start = datetime.date(2015, 1, 1)
end = datetime.date.today()
ticker = '^IXIC'

data = web.DataReader(ticker, 'yahoo', start, end)

close = data[['Adj Close']]
mavg_1 = close.rolling(window = 40).mean()
mavg_2 = close.rolling(window = 100).mean()

#%%
plt.figure(figsize = (20, 10))
plt.plot(close, 'deeppink', linewidth = 3)
plt.plot(mavg_1, 'aqua', linewidth = 2)
plt.plot(mavg_2, 'lime', linewidth = 2)
plt.grid(True)

#%%

