#%%
import pandas_datareader.data as web
import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from IPython.display import Image 
from sklearn.metrics import accuracy_score

# %%

start_date = '2000-1-1'
end_date = '2019-1-31'

stock = ['BAC']
data = web.DataReader(stock, 'yahoo', start_date, end_date, )
print(data.tail())
# %%
# Features construction 
data['Open-Close'] = (data.Open - data.Close)/data.Open
data['High-Low'] = (data.High - data.Low)/data.Low
data['percent_change'] = data['Adj Close'].pct_change()
data['std_5'] = data['percent_change'].rolling(5).std()
data['ret_5'] = data['percent_change'].rolling(5).mean()
data.dropna(inplace=True)

# X is the input variable
X = data[['Open-Close', 'High-Low', 'std_5', 'ret_5']]

# Y is the target or output variable
y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)

# %%
# Total dataset length
dataset_length = data.shape[0]

# Training dataset length
split = int(dataset_length * 0.75)
split

# %%
# Splitiing the X and y into train and test datasets
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train + Test Dataset
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# %%
#PREDICTING USING CLF
clf = RandomForestClassifier(random_state=5)

model = clf.fit(X_train, y_train)
# %%
#PREDICTION
print('Correct Prediction (%): ', accuracy_score(y_test, model.predict(X_test), normalize=True)*100.0)

# %
from sklearn.metrics import classification_report
report = classification_report(y_test, model.predict(X_test))
print(report)

# %%
#STRATEGY RETURNS

data['strategy_returns'] = data.percent_change.shift(-1) * model.predict(X)

(data.strategy_returns[split:]+1).cumprod().plot()
plt.ylabel('Strategy returns (%)')
plt.show()

# %%
#DAILY RETURNS HISTOGRAM
%matplotlib inline
import matplotlib.pyplot as plt
data.strategy_returns[split:].hist()
plt.xlabel('Strategy returns (%)')
plt.show()

# %%
