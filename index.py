
# coding: utf-8

# In[38]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class stockindex:
    
    stocklist=['000001.XSHE','000002.XSHE','000005.XSHE','000004.XSHE','000006.XSHE']  #股票列表
    datebegin=0
    dateindex=0
    jqs=0
    
    def __init__(self,stocklist,datebegin):                      #计算指数基期的市值 
        self.stocklist=stocklist
        jqc=get_price(self.stocklist, start_date= datebegin,  end_date=datebegin, fields='close')   #基期股票价格
        gb=get_shares(self.stocklist, datebegin,datebegin,['total'])                            #基期股本                                                 
        jq=jqc*gb                                                                     #基期股票市值 
        self.jqs=jq.sum(1)                                           #基期股票总市值   参数1意思是对行求和.0是对列求和 
        self.dateindex=jqc.index 
       

    #stockclose=get_price(stocklist, start_date='2005-01-04', end_date='2016-01-04', fields='close')   #日期索引，可以设置指数的时间段
    #dateindex=stockclose.index
    #zs=[]
    def stockdata(self,datastart,dateend,frequency):                     #计算指定日期的值
        a=get_price(self.stocklist, start_date=datastart, end_date=dateend, fields='close',frequency=frequency)    #拿当天股票价格
        b=get_shares(self.stocklist,datastart,dateend,['total'])                        #拿当天股票股本
        size = int(a.size/b.size)
        bnew=np.tile(b,size).reshape(-1,len(self.stocklist))
        s=a*bnew                                                                                   #股票当天市值
       
        zs=(s.sum(1)/self.jqs[0])*1000   #为什么除数直接用jq.sum(1)会失败，因为没有传入这个参数吗  不是！ 是因为两个都带了索引
        #     print (s.sum(1),'kankanwo')
        #  print (s)
        return zs
      
    # stockdata(stocklist,dateindex[5],zsjq(stocklist,'2005-01-04'))       

hihi=stockindex(['000001.XSHE','000002.XSHE','000005.XSHE','000004.XSHE'],'20100104')
# print ("123456")
res = hihi.stockdata("20100101","20100110","10m")
res.plot()
plt.plot(res.values)


# In[40]:

print(res)


# In[13]:

get_price('000001.XSHE', start_date= "20080104",  end_date="20080107", fields='close',frequency = '30m')


# In[ ]:



