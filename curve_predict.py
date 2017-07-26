
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from kshape import kshape, zscore
from scipy import ndimage
from sklearn.cluster import KMeans 
import matplotlib.pyplot

def kshape_cluster(data,idx,idy,plot = True):
    cluster_num = idx * idy
    data = zscore(data,axis = 1)
    cluster = kshape(data,cluster_num)
    predict = pred(cluster,data)
    if plot:
        plot_kshapes(cluster,data,idx,idy)
    return predict
    

def moving_average(data,N,axis = -1):
    return ndimage.convolve1d(data,np.ones(N,)/N,axis)

def plot_kshapes(clusters,data,idx,idy):
    fig,ax = pyplot.subplots(idx,idy)
    cluster_num = idx * idy
    for j in range(cluster_num):
        if j < cluster_num:
            for i in clusters[j][1]:
                ax[int(j/idy),j%idy].plot(data[i,:])
    fig.set_figheight(30)
    fig.set_figwidth(45)
    pyplot.show()
    
def pred(clusters,data):
    pred = np.empty(data.shape[0])
    count = 0
    for centroid,cluster in clusters:
        for id in cluster:
            pred[id] = count
        count = count + 1
    return pred

def plot_kmeans(pred,data,idx,idy):
    fig,ax = pyplot.subplots(idx,idy)
    cluster_num = idx * idy
    count = 0
    for i in pred:
        ax[int(i/idy),i%idy].plot(data[count,:])
        count = count + 1
    fig.set_figheight(30)
    fig.set_figwidth(45)
    pyplot.show()
    
def kmeans_cluster(data,idx,idy,plot = True):
    data = zscore(data,axis=1)
    clf = KMeans(n_clusters=idx * idy,max_iter = 30000,init='k-means++') 
    y_pred = clf.fit_predict(data)
    if plot:
        plot_kmeans(y_pred,data,idx,idy)
    return y_pred
    

def calc_y(i):
    if i > 0:
        return 1
    return 0

def profit(predicted,y):
    pred_ones = np.count_nonzero(predicted)
    positive = 1. * np.count_nonzero(predicted * y)
    if pred_ones == 0:
        return 0
    return positive / pred_ones


# In[2]:

code = '沪深300'
# code = '000521.XSHE'
start_date = '2012-01-02'
end_date = '2017-6-15'

m1=get_price(code,start_date=start_date,end_date=end_date,frequency='1m')
d1=get_price(code,start_date=start_date,end_date=end_date,frequency='1d').as_matrix()

open_m1 = np.array(m1['open'])

close_m1 = np.array(m1['close'])
close_d1 = close_m1.reshape(-1,240)[:,-1]
close_d1_yes = np.roll(close_d1,1)
close_d1_yes[0] = open_m1[0]

open_d1 = open_m1.reshape(-1,240)[:,0]
open_d1_tom = np.roll(open_d1,-1)
open_d1_tom[-1] = close_d1[-1]

open_m1 = open_m1.reshape(-1,240)

open_120 = open_m1[:,119]

open_m1 = np.transpose((open_m1.transpose() / close_d1_yes) - 1)

TR = np.amax(np.array([d1[:,2] - d1[:,3],abs(np.roll(d1[:,1],1) - d1[:,3]),abs(d1[:,2] - np.roll(d1[:,1],1))]),axis = 0) / close_d1_yes
TR[0] = (d1[0,2] - d1[0,3]) / close_d1_yes[0]

mask = (TR>0.03)

open_m1_60 = open_m1[:,:59]
open_m1_120 = open_m1[:,:119]
open_m1_60_120 = open_m1[:,60:119]

rate = open_d1_tom / open_120 - 1
y = np.array(list(map(calc_y,rate)))

# open_m1 = open_m1[mask]
# open_m1_60 = open_m1_60[mask]
# open_m1_120 = open_m1_120[mask]
# open_m1_60_120 = open_m1_60_120[mask]
# y = y[mask]


# In[3]:

from sklearn.svm import SVC
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm, grid_search

idx = 3
idy = 3
kmeans_pred = kmeans_cluster(open_m1,idx,idy,False)
# kmeans_pred_60 = kmeans_cluster(open_m1_60,idx,idy,False)
kmeans_pred_120 = kmeans_cluster(open_m1_120,idx,idy,False)
# kmeans_pred_60_120 = kmeans_cluster(open_m1_60_120,idx,idy,False)

X = np.hstack((kmeans_pred.reshape(-1,1),kmeans_pred_120.reshape(-1,1)))
X = np.nan_to_num(X)

X = X[1:,:]
y = y[1:]

train_num = int(0.7 * X.shape[0])
test_num = X.shape[0] - train_num

# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : gammas}
# grid_search = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
# grid_search.fit(X[:train_num,:],y[:train_num])
# predicted = grid_search.predict(X[-test_num:,:])
# print(grid_search.best_params_)

# clf = SVC(kernel="rbf", C=2.8, gamma=.0073, probability=True,class_weight = {1:2,0:1})
clf = SVC(kernel="rbf", C=1, gamma=.0073, probability=True)

clf.fit(X[:train_num,:],y[:train_num])
predicted = clf.predict(X[-test_num:,:])
print("Accuracy: %0.4f" % metrics.accuracy_score(y[-test_num:],
                                                     predicted))


# In[4]:

# plt.scatter(range(len(kmeans_pred_120[1:])),kmeans_pred_120[1:],c = y )

print(profit(predicted,y[-test_num:]))
print(predicted)
print(y[-test_num:])


# In[5]:

plt.hist(TR,10,(0,0.1))


# In[6]:

import tushare as ts

data = ts.get_k_data('000300', index=True,start='2016-10-01', end='2016-10-31')


# In[ ]:



