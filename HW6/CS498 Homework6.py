# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:37:10 2017

@author: Xinchen
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from collections import Counter


os.chdir("C:/Users/Xinchen/Desktop/cs498/hw6")

filename1 = "docword.nips.txt"

filename2 = "vocab.nips.txt"

with open(filename1) as f:
    data = f.read().split("\n")   # original data file
    
with open(filename2) as g:
    vocab = g.read().split("\n")  # vocabulary file

data1 = data[3:len(data)-1]       # remove the first three lines and last line
    
dt = np.zeros((1500, 12419))

for i in range(np.shape(data1)[0]):                   # make the data matrix
       docID = int(data1[i].split()[0]) - 1 
       wordID = int(data1[i].split()[1]) - 1
       count = float(data1[i].split()[2])
  
       dt[docID, wordID] = count

W = np.zeros((1500, 30))
topic = 30
pi_j = np.zeros(topic)

word = np.shape(dt)[1] #12419

document = np.shape(dt)[0] #1500

kmeans1 = KMeans(n_clusters =topic, random_state=0).fit(dt)  ## fit the k-means model

for i in range(topic):
   pi_j[i] = Counter(kmeans1.labels_)[i] / dt.shape[0]
 
p_jk_temp = np.zeros((topic,  np.shape(dt)[1]))
p_jk = np.zeros((topic,  np.shape(dt)[1]))
for i, label in enumerate(kmeans1.labels_):
     for j in range(topic):
               if label == j:
                    p_jk_temp[j, ] += dt[i,]
for i in range(topic):
     for j in range(np.shape(dt)[1]):
           if p_jk_temp[i,j] == 0:
                p_jk_temp[i,j] += 1
                         
p_jk = p_jk_temp / np.sum(p_jk_temp, axis = 1)[:, None]
     

W1_nom = np.zeros((document, topic))
temp1 = np.zeros((document, topic))
Z = np.zeros((document, np.shape(dt)[1]))

############### E-step##################
QQ1 = []  #Empty list to hold Q
QQ1.append(0) #First value of Q
for k in range(1000):
    for i in range(document):
        for j in range(topic):
            temp1[i, j] = np.sum(dt[i,] * np.log(p_jk[j,])) + np.log(pi_j[j])
   
        Max = temp1.max(1)
        temp2 = np.exp(temp1 - Max[::,None])
        W = temp2 / temp2.sum(1)[::,None]
           
    ############M-step##################
    p_jk_nom = np.zeros((topic, dt.shape[1]))
    for i in range(topic):
         for j in range(document):
             p_jk_nom[i, ] += np.dot(dt[j,], W[j,i])
    p_jk_nom += 0.00001    # prevent 0 word probability
    p_jk_denom = np.sum(p_jk_nom, axis = 1) + 0.00001*dt.shape[1] 
    p_jk = p_jk_nom / p_jk_denom[::, np.newaxis]

    pi_j = np.sum(W, axis = 1) / document

##############Convergnce##############
    Q1 = 0
    for i in range(document):
        for j in range(topic):
           Q1 += (np.dot(dt[i,], np.log(p_jk[j,:])) + \
                  np.log(pi_j[j])) * W[i,j]
    QQ1.append(Q1)

  
# check convergence
    if np.abs(QQ1[k] - QQ1[k-1]) < 0.0001:  
          break
     
#######Get the probability with with the topic is selected
topic_index = np.zeros(document)
prob = np.zeros(30)
for i in range(document):    
        topic_index[i] = np.argsort(W[i,])[::-1][0]
        for j in range(topic):
           if topic_index[i] == j:
                  prob[j] +=  1
prob = prob / document
ax, fig =  plt.subplots()  
plt.plot([i for i in range(1,31)], prob)
plt.xlabel('Topic')
plt.ylabel('Probability') 
plt.title("Probability Chart")
plt.show()      
#######Get the most frequent 10 words for the 30 topics

Vocab = vocab[0:len(vocab) - 1]
freq_word_index = []
freq_word = np.zeros((topic, 10))

for i in range(topic):
    freq_word_index.append(np.argsort(p_jk[i,])[::-1][0:10])

freq_word = np.zeros((30,10), dtype = object)

for i in range(len(freq_word_index)): #30
    for j in range(10):
             freq_word[i][j]= vocab[freq_word_index[i][j]]

df = pd.DataFrame(freq_word)
df.to_csv("freq_word.csv")


#####Image segmentation using EM #####
from PIL import Image
from copy import deepcopy

def getPixel(pic):  #One method to construct pixel data from pictures

    r = np.zeros((pic.size[0], pic.size[1]))
    g = np.zeros((pic.size[0], pic.size[1]))
    b = np.zeros((pic.size[0], pic.size[1]))
    
    for i in range(pic.size[0]):
        for j in range(pic.size[1]):
            r[i,j] = pic.getpixel((i,j))[0]
            g[i,j] = pic.getpixel((i,j))[1]
            b[i,j] = pic.getpixel((i,j))[2]
            
        Pixel_data = np.transpose(np.array((r.ravel(), g.ravel(), b.ravel())))
        Pixel_data_scaled = (Pixel_data - np.mean(Pixel_data, axis = 0)) / np.std(Pixel_data, axis = 0)
    return Pixel_data, Pixel_data_scaled  # scale the data

def getPixel2(pic): #Another  method to construct pixel data from pictures
      rgb = np.zeros((pic.size[0]*pic.size[1], 3))
      for k in range(rgb.shape[0]):
           i = k // pic.size[1]
           j = k % pic.size[1]
           rgb[k,] = pic.getpixel((i,j)) 
      return rgb 

def generate_Image(pic, segment, seed = 0):
     
     width = pic.size[0]
     height = pic.size[1]
     
     Pixel = getPixel(pic)[1]

     pi_j2 = np.zeros(segment)

     kmeans2 = KMeans(n_clusters=segment, random_state=seed).fit(Pixel) # Kmeans model

     for i in range(segment):
          pi_j2[i] = Counter(kmeans2.labels_)[i] / Pixel.shape[0] # Get initial probabilities

     mu = kmeans2.cluster_centers_ # Get the initial values for mu

     
     W2_nom = np.zeros((Pixel.shape[0], segment)) 
 ##########################E-step#####################
     QQ2 = []
     QQ2.append(0)
 
     for k in range(0, 100):
          for i in range(Pixel.shape[0]):
               for j in range(segment):
                    W2_nom[i,j] = np.exp(-0.5*(np.dot((Pixel[i,] - mu[j,]), \
                          (Pixel[i,] - mu[j,])))) * pi_j2[j]

          W2_denom = np.sum(W2_nom, axis = 1)  

          W2 = W2_nom / W2_denom[::, None]
     
 ##########################M-step#####################
          mu_denom = np.sum(W2, axis = 0)
          mu_nom = np.zeros((segment, 3))
    
          for m in range(segment):
              for n in range(Pixel.shape[0]):
                   mu_nom[m,] += Pixel[n,] * W2[n,m]
             
          mu = mu_nom / mu_denom[::, None]             # update mu
 
          pi_j2 = np.sum(W2, axis = 0) / Pixel.shape[0]  #update pi
    
###################Calculate Q#######################
          Q2 = 0
          for i in range(Pixel.shape[0]):
              for j in range(segment):
                   Q2 += (-0.5*(np.dot((Pixel[i,] - mu[j,:]),(Pixel[i,] - mu[j,:]))) + \
                         np.log(pi_j2[j])) * W2[i,j]
   
          QQ2.append(Q2)
          
          if np.abs(QQ2[k] - QQ2[k-1]) < 0.0001:  # check convergence
                  break
          
###Map each pixel to the cluster center
###with the highest value of the posterior probability for that pixel
          
     segment_index = []
     for i in range(Pixel.shape[0]):   # Get the cluster
          segment_index.append(np.argsort(W2[i,])[segment - 1])
    
     Pixel_new = deepcopy(Pixel)

     for i in range(Pixel.shape[0]):
          Pixel_new[i,] = mu[segment_index[i],]
 
     Pixel_back = np.zeros((width, height, 3))  

     for k in range(Pixel.shape[0]): # transform the data back to original dimension
              i = k // height
              j = k % height
              Pixel_back[i,j,] = Pixel_new[k,]
     
     Pixel_final_mirror = deepcopy(Pixel_back) 

     for i in range(width):     #mirror the image
          for j in range(height):
                Pixel_final_mirror[i, j,] = Pixel_back[i, height - 1 - j,]

     Pixel_final_rotate = np.zeros((height, width, 3))
     for i in range(width):    #rotate the image     
          for j in range(height):
                Pixel_final_rotate[j, i, ] = Pixel_final_mirror[i, height -1 - j, ]   

     Pixel_scaled_back =  (Pixel_final_rotate * getPixel(pic)[0].std(0)) + getPixel(pic)[0].mean(0)
    
     img = Image.fromarray(Pixel_scaled_back.astype(np.uint8), "RGB")
     img.show()
     if height == 330:
          img.save("smallsunset" + str(seed) + "_" + str(segment) + "segments.jpg")
     elif height == 399:
          img.save("smallstrelitzia" + "_" + str(segment) + "segments.jpg")
     else:
          img.save("RobertMixed03" + "_" + str(segment) + "segments.jpg")
     return img
######################Display Image#######################

pic1 = Image.open("smallsunset.jpg")
pic2 = Image.open("smallstrelitzia.jpg")
pic3 = Image.open("RobertMixed03.jpg")

#smallsunset.jpg
generate_Image(pic1, segment = 10)  
 #smallsunset.jpg (five different start points)
seed = np.random.randint(100, size=5)
for i in range(len(seed)):
     generate_Image(pic1, segment = 20, seed = seed[i])

generate_Image(pic1, segment = 50)

#smallstrelitzia.jpg
generate_Image(pic2, segment = 10)  
generate_Image(pic2, segment = 20)
generate_Image(pic2, segment = 50)

#RobertMixed03.jpg
generate_Image(pic3, segment = 10)  
generate_Image(pic3, segment = 20)
generate_Image(pic3, segment = 50)



