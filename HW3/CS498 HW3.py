# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:05:06 2017

@author: Xinchen Pan, Xiaozhifang Yu, Jiefei Shi
"""
import numpy as np      # Load libraries we need
import pandas as pd
from sklearn.decomposition import PCA
import os 
import matplotlib.pyplot as plt

os.chdir("C:/Users/Xinchen/Desktop/cs498/hw3")

################4.1（a)######################

##############Data Preparation##############

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

data = [] #make an empty list

for i in range(len(pickle_data)):         #load the data 
    data.append(unpickle(pickle_data[i]))


def load_dt(data):                         #Separate the label and predictors
    for i in range(len(data)):
        if i == 0:
            X = data[i]['data']
            Y = data[i]['labels']
        else:
            X = np.concatenate((X, data[i]['data']))
            Y = np.concatenate((Y, data[i]['labels']))
    return X,Y

X, Y = load_dt(data)  # Separated

X = X/255 # rescale X 

dt = np.column_stack((X, Y))  #combine
##############get x_i-mean({x) and separate the data by category#############

#category mean 
cm = pd.DataFrame(dt[:,0:3072]).groupby(pd.Series(dt[:,3072]), axis=0).mean()

zero_mean = []  #Not necessary if we use transform() from library
category = []

for i in range(10):  #We get X-mean(x) from the following code
    category.append(X[Y == i])
    zero_mean.append(category[i] - np.array(cm.iloc[i]))
 
cm = np.asarray(cm)

##############Get Error for each image######################
error = []
error_unscaled = []
pca20 = []
ncomp = 20

xhat_list = []
for i in range(10):
    pca = PCA()
    pca.fit(category[i])
    pca20.append(pca.components_[:ncomp,:])
    xhat_list.append(np.dot(pca.transform(category[i])[:,:ncomp], pca.components_[:ncomp,:])) 
    xhat_list[i] = np.add(xhat_list[i], cm[i,:])
    error.append(np.sum(np.square(xhat_list[i] - category[i])))
    error_unscaled.append(np.sum(np.square(255*(xhat_list[i] - category[i]))))

###############Plot########################

label = ["airplain", "automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
error_array = np.asarray(error)

fig, ax = plt.subplots()
ax.scatter(np.arange(10),error_array)

for i, txt in enumerate(label):
    ax.annotate(txt, (np.arange(10)[i],error_array[i]))

plt.title("Category Error")
plt.show()
###################(b)######################
#distance matrix#

dist = np.zeros((10,10))

for i in range(10):
   for j in range(i+1,10):
       dist[i,j] = np.linalg.norm((cm[i,:] - cm[j,:]))
       dist[j,i] = dist[i,j]
####Prinipal Coordinate Analysis###

A = np.identity(10) - np.reshape(np.ones(100)/10,(10,10))

W = -0.5*np.dot(np.dot(A.T,dist),A)

eigenVa, eigenVe= np.linalg.eig(W)

idx = eigenVa.argsort()[::-1]   
eigenVa = eigenVa[idx]  #Sort eigenvalue
eigenVe = eigenVe[:,idx]
Lambda = np.dot(np.dot(eigenVe.T, W), eigenVe)


lambda2 = Lambda[0:2,0:2]
lambda2[0,0] = np.sqrt(lambda2[0,0])  #Take square root
lambda2[1,1] = np.sqrt(lambda2[1,1])


V = np.dot(lambda2 , eigenVe[:,0:2].T) 


###############Plot########################
fig, ax = plt.subplots()
ax.scatter(V.T[:,0],V.T[:,1])
plt.title("Principal Coordinate Analysis(1)")

for i, txt in enumerate(label):
    ax.annotate(txt, (V.T[i,0],V.T[i,1]))
plt.show()

################(c)##################

error_simi1 = np.zeros((10,10))
error_simi2 = np.zeros((10,10))

######################E(B to A)#################
for i in range(9):
    pca3 = PCA(n_components=20)
    pca3.fit(category[i])
    for j in range(i+1, 10):
        xhat_temp = np.dot(pca3.transform(category[i]), pca3.components_)
        xhat_temp = np.add(xhat_temp, cm[j,:])
        error_simi1[i,j] =  np.sum(np.square(xhat_temp - category[i]))
        error_simi1[j,i] =  error_simi1[i,j] 

################E(A to B)####################
for i in range(9):
    pca3 = PCA(n_components=20)
    pca3.fit(category[i+1])
    for j in range(i+1, 10):
        xhat_temp = np.dot(pca3.transform(category[j]), pca3.components_)
        xhat_temp = np.add(xhat_temp, cm[i,:])
        error_simi2[i,j] =  np.sum(np.square(xhat_temp - category[j]))
        error_simi2[j,i] =  error_simi2[i,j]
     
error_simi = 0.5*(error_simi1 + error_simi2)

################Follow the same step in (b)######################

A = np.identity(10) - np.reshape(np.ones(100)/10,(10,10))

W2 = -0.5*np.dot(np.dot(A.T, error_simi),A)

eigenVa1, eigenVe1= np.linalg.eig(W2)

idx = eigenVa1.argsort()[::-1]   
eigenVa1 = eigenVa1[idx]
eigenVe1 = eigenVe1[:,idx]
Lambda2 = np.dot(np.dot(eigenVe1.T, W2), eigenVe1)


lambda3 = Lambda2[0:2,0:2]
lambda3[0,0] = np.sqrt(lambda3[0,0])
lambda3[1,1] = np.sqrt(lambda3[1,1])

V2 = np.dot(lambda3 , eigenVe1[:,0:2].T)

###############Plot########################
fig, ax = plt.subplots()
ax.scatter(V2.T[:,0],V2.T[:,1])
plt.title("Principal Coordinate Analysis(2)")

for i, txt in enumerate(label):
    ax.annotate(txt, (V2.T[i,0],V2.T[i,1]))
plt.show()
