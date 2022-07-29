#%%
import pandas as pd
import pandas_datareader as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")

#%%
start = datetime.date(2010, 1, 4)
end = datetime.date.today()
ticker = '^IXIC'

data = web.DataReader(ticker, 'yahoo', start, end)
close = data[['Adj Close']]

#%%
plt.figure(figsize = (20, 10))
plt.plot(close)
plt.grid(True)


#%%


#%%