
# coding: utf-8

# In[51]:

import re

freq_int = {
    '1m':1,
    '5m':2,
    '10m':3,
    '15m':4,
    '30m':5,
    '60m':6,
    '1d':7
}

###公司mysql数据库格式
def get_future_price1(code,start_date,end_date,frequency):
    df = get_price(code.upper(),start_date=start_date,end_date=end_date,frequency=frequency)
    df['dateInt'] = [ int(i.strftime("%Y%m%d")) for i in df.index ]
    df['timeInt'] = [ int(i.strftime("%H%M%S")) for i in df.index ]
    df['realDate'] = range(df.shape[0])
    df['vol'] = df['volume']
    df['openInt'] = df['open_interest']
    df['contract'] = re.match('[a-zA-Z]+',code).group()
    df['cycle'] = freq_int[frequency]
    del df['total_turnover']
    del df['volume']
    del df['trading_date']
    del df['limit_up']
    del df['limit_down']
    del df['basis_spread']
    del df['open_interest']
    return df

###vnpy回测mongodb格式
def get_future_price2(code,start_date,end_date,frequency):
    contract = re.match('[a-zA-Z]+',code).group()
    contract = contract + '0000'
    df = get_price(code.upper(),start_date=start_date,end_date=end_date,frequency=frequency)
    df['vtSymbol'] = contract
    df['symbol'] = contract
    df['date'] = [ int(i.strftime("%Y%m%d")) for i in df.index ]
    df['time'] = [ int(i.strftime("%H%M%S")) for i in df.index ]
    df['datetime'] = df.index
    df['openInterest'] = df.open_interest

    del df['total_turnover']
    del df['trading_date']
    del df['limit_up']
    del df['limit_down']
    del df['basis_spread']
    del df['open_interest']
    return df,contract

###插入到vnpy回测数据库中
def insert_mongo2(db,code,start_date,end_date,frequency):
    df,collection = get_future_price2(code,start_date,end_date,frequency)
    db[collection].insert_many(df.to_dict('record'))


# In[52]:

from pymongo import MongoClient

client = MongoClient('blabla', 666666)
db = client['blabla']

code = 'rb88'
start_date = '20170101'
end_date = '20170714'
frequency = '1m'

insert_mongo2(db,code,start_date,end_date,frequency)


# In[32]:

import tushare as ts

df = ts.get_realtime_quotes('000581')


# In[35]:

df = ts.get_today_ticks('601333')
df.tail(10)


# In[36]:

print(df)


# In[ ]:



