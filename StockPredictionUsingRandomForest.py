#%%
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('dark_background')
#%%
start = dt.datetime(2010, 1, 1)

end = dt.datetime.today()

data = web.DataReader('^IXIC', data_source =
                      'yahoo', start = start, end = end)

close_data = data.loc[:, 'Adj Close']
#%%
daily_pct_change = close_data.pct_change()

daily_pct_change.fillna(0, inplace=True)

daily_pct_change.hist(bins = 100)

rolling_avg = close_data.rolling(window = 20).mean()

#%%
plt.title("Historical Data")

close_data.plot(figsize = [12, 6], color = 'Turquoise')

rolling_avg.plot(color='red')

#plt.savefig('C:/Users/chaua/OneDrive/Documents/Advik/Python/snp_historical.png', dpi=600)

#%%
data_shifted = data.shift(1)
data_shifted = data_shifted[1:-1]

data = data[1:-1]

print(data_shifted, data)

#%%
#train-test split

X = data_shifted[['Close']]
y = data['Adj Close']

X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=1)
model = RandomForestRegressor(n_estimators = 300
                                  , criterion = 'mse',  random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_future = model.predict(X_test)


results = metrics.mean_absolute_error(y_test, y_pred)

print(y_test, y_pred)
print("Mean Absolute Error:", results)




