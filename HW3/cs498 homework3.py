# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:49:01 2017

@author: Xinchen
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict
    
batch1 = "cifar-10-batches-py/data_batch_1"
batch2 = "cifar-10-batches-py/data_batch_2"
batch3 = "cifar-10-batches-py/data_batch_3"
batch4 = "cifar-10-batches-py/data_batch_4"
batch5 = "cifar-10-batches-py/data_batch_5"
test_batch = "cifar-10-batches-py/test_batch"


pickle_data = [batch1, batch2, batch3, batch4, batch5, test_batch]

data = []

for i in range(len(pickle_data)):
    data.append(unpickle(pickle_data[i]))


def load_dt(data):
    for i in range(len(data)):
        if i == 0:
            X = data[i]['data']
            Y = data[i]['labels']
        else:
            X = np.concatenate((X, data[i]['data']))
            Y = np.concatenate((Y, data[i]['labels']))
    return X,Y

X, Y = load_dt(data)

X = X/255

dt = np.column_stack((X, Y))

    
#category mean
cm = pd.DataFrame(dt[:,0:3072]).groupby(pd.Series(dt[:,3072]), axis=0).mean()


category = []
zero_mean = []

for i in range(10):
    category.append(X[Y == i])
    zero_mean.append(category[i] - np.array(cm.iloc[i]))

cov_m = []
eigenVa = []
eigenVe = []

for i in range(10):
    cov_m.append((zero_mean[i].T.dot(zero_mean[i]))/(6000 - 1)) 
    eigenVaV, eigenVeV = np.linalg.eig(cov_m[i])
    eigenVa.append(eigenVaV)
    eigenVe.append(eigenVeV)

   
result_comp = []
for i in range(10):
    result_comp.append(eigenVe[i][:,0:20].T)

r = []         
for i in range(10):
   r.append(eigenVe[i].T.dot(zero_mean[i].T))

eigenVe20 = eigenVe[1:20]

pca20 = []

for i in 1:range(10){
    pca = PCA(n_components=20)
    pca.fit(category[i])
    pca20.append(pca.components_)

    ################test#####################
    
pca = PCA(n_components = 2)
pca.fit(X)
r =  np.dot(pca.components_, (X - mu).T) #Form the data set r
temp = r[0:2,]  #Took 20 
temp0 = np.zeros((2,150))
p = np.row_stack((temp, temp0))
xhat = np.dot(pca.components_, p)                
xhat = np.add(xhat.T, mu)
error = np.sum(np.square(xhat - category[0]))