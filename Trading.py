#%%
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
#%%
data = pd.read_excel('BB.xlsx')
data = data.dropna()
data.head
data.columns

# %%
position = 0
stop_loss = 0.5
cump = 0
notional = 1000000
trade_info = []
exit_val = None
pnl = None
exit_pos = False

for row in range(20, len(data)):
    close = data['Close'].iloc[row]
    bb_upper = data['BB_UPPER'].iloc[row]
    if position ==0:
        if close > bb_upper:
            position = -1
            entry_val = close
        
        elif close < bb_upper:
            position = 1  
            entry_val = close

    else:
        if position == 1:
            if close > bb_upper:
                position = -1  #Change from buy to sell
                pnl = (((close-entry_val)/entry_val))*notional
                exit_val = close
                cump +=pnl
                exit_pos = True
            elif ((entry_val-close)/entry_val)*100 > stop_loss:
                position = 0
                pnl = (((close-entry_val)/entry_val))*notional
                exit_val = close
                cump +=pnl
                exit_pos = True
        elif position == -1:
            if close < bb_upper:
                position = 1
                pnl = (((entry_val-close)/entry_val))*notional
                exit_val = close
                cump +=pnl
                exit_pos = True
            elif ((close-entry_val)/entry_val)*100 > stop_loss:
                position = 0
                pnl = (((entry_val-close)/entry_val))*notional
                exit_val = close
                cump +=pnl
                exit_pos = True
    if exit_pos ==  True:
        Trade_Data = {'Trade Date': data['Dates'].iloc[row], 'Position': position, 'Entry': entry_val, 'Exit': exit_val, 'PNL': pnl, 'CUM_PNL': cump }
        trade_info.append(Trade_Data)

trade_info = pd.DataFrame(trade_info, columns = ['Trade Date','Position', 'Entry', 'Exit', 'PNL', 'CUM_PNL'])

#data['Market Return'] = np.log(close / close.shift(1))

#data['Strategy Return'] = data['Market Return'] * data['Position']

#data['Strategy Return'].cumsum().plot()

#data[['Position', 'Close']].plot(figsize=(12,6))



# %%

trade_info.to_excel('Trade_Info.xlsx')
# %%
