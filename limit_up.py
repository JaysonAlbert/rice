
# coding: utf-8

# In[57]:

import numpy as np
import matplotlib.pyplot as plt
import math

def plot_open_volume(ax,df,title):
    ax2 = ax.twinx()
    ax.plot(df[:,0],'r')
    ax2.plot(df[:,1],'b')
    ax.set_title(title)
    ax.set_ylabel('open')
    ax2.set_ylabel('volume')
    
def get_1m_data(code,date):
    return get_price(code,start_date=date[0],end_date=date[1],fields = ['open','volume'],frequency='1m').as_matrix()

code = '000521.XSHE'
start_date = '2012-01-02'
end_date = '2017-6-26'

d1=get_price(code,start_date=start_date,end_date=end_date,fields = ['open','close','limit_up'],frequency='1d')
limit_up = (d1[d1['close'] == d1['limit_up']].index)

next_day = [(date,get_next_trading_date(date)) for date in limit_up]

length = math.ceil(math.sqrt(len(next_day)))

count = 0
fig, ax = plt.subplots(length,height)
for date in next_day:
    plot_open_volume(ax[math.floor(count/length),count % length],get_1m_data(code,date),date[0])
    count = count + 1l
fig.set_figwidth(length * 15)
fig.set_figheight(40)


# In[1]:

import plotly.offline as py
import plotly.graph_objs as go
import plotly

plotly.offline.init_notebook_mode()
code = '000521.XSHE'
start_date = '2017-01-02'
end_date = '2017-6-26'
df=get_price(code,start_date=start_date,end_date=end_date,frequency='1d')
trace = go.Candlestick(x=df.index,
                       open=df.open,
                       high=df.high,
                       low=df.low,
                       close=df.close)

data = [trace]
py.iplot(data)


# In[7]:

df['mean_open'] = df['open'].rolling(2).mean()
del df['total_turnover']
del df['volume']

print(df)


# In[ ]:



