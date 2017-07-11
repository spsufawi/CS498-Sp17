# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:08:31 2017

@author: Xinchen Pan, Jiefei Shi, fangxiaozhi Yu
"""

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
from numpy import zeros, float32
from copy import deepcopy
from pylab import imshow, show, cm


os.chdir("C:/Users/Xinchen/Desktop/cs498/hw7")

##Loaded in the dataset based on the tutorial
##https://martin-thoma.com/classify-mnist-with-pybrain/

images = "train-images-idx3-ubyte.gz"

def get_labeled_data(imagefile):

    images = gzip.open(imagefile, 'rb')

    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    x = zeros((500, rows, cols), dtype=float32)  
    for i in range(500):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
    return x


dt = get_labeled_data(images) / 255
pixel = deepcopy(dt)

#####binarize#####
for k in range((500)):
    for i in range(pixel.shape[1]):
         for j in range(pixel.shape[2]):
               if pixel[k][i][j] <= 0.5:
                    pixel[k][i][j] = -1
               else:
                    pixel[k][i][j] = 1
                         
noise_pixel = deepcopy(pixel)

shape = pixel.shape[1] #28

#####Add noise#####
for k in range((500)):
     for i in range(shape):
          for j in range(shape):
               if np.random.binomial(1,0.98) == 1:
                    noise_pixel[k][i,j] = pixel[k][i,j]
               else:
                     noise_pixel[k][i,j] = -pixel[k][i,j]
                    
deno_image =  np.zeros((500, pixel.shape[1], pixel.shape[2]))


Q = [[] for x in range(500)]
for i in range(500):
     Q[i].append(0)
    
#####denoise algorithm#####
for k in range(500):
     pi_j = np.ones((shape,shape))/2
     pi_j_prev = np.ones((shape,shape))/2
     for m in range(2,10):
          upper_n1 = np.zeros((shape,shape))
          upper_p1 = np.zeros((shape,shape))
          for i in range(shape):
               for j in range(shape):  #####four neighbours#####
                    if (i in range(1,shape-1) and j in range(1,shape-1)): 
                      upper_p1[i,j] = 0.2*(2*pi_j[i][j+1]-1 + 2*pi_j[i][j-1]-1+ \
                                 (2*pi_j[i-1][j] -1)+ 2* pi_j[i+1][j]-1) + \
                                 2*(noise_pixel[k][i][j+1] + noise_pixel[k][i][j-1] + \
                                    noise_pixel[k][i-1][j] + noise_pixel[k][i+1][j])
                      upper_n1[i,j] = -0.2*(2*pi_j[i][j+1]-1 + 2*pi_j[i][j-1]-1+ \
                                 (2*pi_j[i-1][j] -1)+ 2* pi_j[i+1][j]-1) -\
                                 2*(noise_pixel[k][i][j+1] + noise_pixel[k][i][j-1] + \
                                    noise_pixel[k][i-1][j] + noise_pixel[k][i+1][j])  
                    
                    elif i == 0 and j == 0: ##two neighbours##
                          upper_p1[i,j] = 0.2*((2*pi_j[i][j+1]-1) + \
                                 2*pi_j[i+1][j]-1) + \
                                 2*(noise_pixel[k][i][j+1] + \
                                    + noise_pixel[k][i+1][j])
                          upper_n1[i,j] = -0.2*((2*pi_j[i][j+1]-1) + \
                                 2*pi_j[i+1][j]-1) - \
                                 2*(noise_pixel[k][i][j+1] + \
                                    + noise_pixel[k][i+1][j])
                                 
                    elif i == 0 and j == shape-1:
                         upper_p1[i,j] = 0.2*(2*pi_j[i][j-1]-1 + \
                                 2*pi_j[i+1][j]-1) + \
                                 2*(noise_pixel[k][i][j-1] + \
                                    + noise_pixel[k][i+1][j])
                                 
                         upper_n1[i,j] = -0.2*(2*pi_j[i][j-1]-1 + \
                                 2*pi_j[i+1][j]-1) - \
                                 2*(noise_pixel[k][i][j-1] + \
                                    + noise_pixel[k][i+1][j])
                                                    
                    elif i == shape-1 and j == 0:
                         upper_p1[i,j] = 0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j+1]-1) + \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j+1])
                         upper_n1[i,j] = -0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j+1]-1) - \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j+1])                   
                                 
                    elif i == shape-1 and j == shape-1:
                         upper_p1[i,j] = 0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j-1] -1) + \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j-1])
                         upper_n1[i,j] = -0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j-1] -1) - \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j-1])
                    
                    elif j == 0:  ##three neighbours##
                         upper_p1[i,j] = 0.2*(2*pi_j[i-1][j] -1 + \
                                 2*pi_j[i][j+1] -1 +2*pi_j[i+1][j] -1) + \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j+1] +\
                                      noise_pixel[k][i+1][j])
                         upper_n1[i,j] = -0.2*(2*pi_j[i-1][j] -1 + \
                                 2*pi_j[i][j+1] -1 +2*pi_j[i+1][j] -1) + \
                                 2*(noise_pixel[k][i-1][j]) - \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j+1] +\
                                      noise_pixel[k][i+1][j])
                    elif i == 0:
                         upper_p1[i,j] = 0.2*(2*pi_j[i+1][j]-1 + \
                                 2*pi_j[i][j+1]-1 +2*pi_j[i][j-1]-1 ) + \
                                 2*(noise_pixel[k][i+1][j] + \
                                    + noise_pixel[k][i][j+1] +\
                                      noise_pixel[k][i][j-1])
                         upper_n1[i,j] = -0.2*(2*pi_j[i+1][j]-1 + \
                                 2*pi_j[i][j+1]-1 +2*pi_j[i][j-1]-1 ) - \
                                 2*(noise_pixel[k][i+1][j] + \
                                    + noise_pixel[k][i][j+1] +\
                                      noise_pixel[k][i][j-1])
                    elif i == shape-1:
                         upper_p1[i,j] = 0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j+1] -1 +2*pi_j[i][j-1] -1) + \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j+1] +\
                                      noise_pixel[k][i][j-1])
                         upper_n1[i,j] = -0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j+1]-1 +2*pi_j[i][j-1] -1) - \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i][j+1] +\
                                      noise_pixel[k][i][j-1])
                    else:
                       upper_p1[i,j] = 0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j-1] -1 +2*pi_j[i+1][j]-1 ) + \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i+1][j] +\
                                      noise_pixel[k][i][j-1])
                                 
                       upper_n1[i,j] = -0.2*(2*pi_j[i-1][j]-1 + \
                                 2*pi_j[i][j-1] -1 +2*pi_j[i+1][j]-1 ) - \
                                 2*(noise_pixel[k][i-1][j] + \
                                    + noise_pixel[k][i+1][j] +\
                                      noise_pixel[k][i][j-1])
             
                    pi_j[i,j] = (np.exp(upper_p1[i,j]) / (np.exp(upper_n1[i,j]) + np.exp(upper_p1[i,j])))
                    
                    if pi_j[i,j] < 0.5:
                         deno_image[k][i][j] = -1
                    else:        
                         deno_image[k][i][j] = 1
          
     
          Q[k].append(np.sum((pi_j - pi_j_prev) ** 2))
          pi_j_prev = deepcopy(pi_j)  
          
          if Q[k][m-1]<  0.000001:
               break

correct_frac = np.zeros(500)
count = np.zeros(500)

####fraction of correct pixels for each image####
for k in range(500):

     for i in range(shape):
          for j in range(shape):
               if deno_image[k][i][j] == pixel[k][i][j]:
                    count[k] += 1
     correct_frac[k] = count[k] / (shape * shape)


plt.plot(range(1,501), correct_frac)
plt.xlabel("Image ID")
plt.ylabel("Accuracy")

def view_image(image, label=""):  #Also from the website where we found the way to load
    """View a single image."""    # the dataset
    imshow(image, cmap=cm.gray)
    show()
    
np.savetxt("correct_frac.csv",correct_frac, delimiter=",")

least_accur = count.argsort()[0]
most_accur = count.argsort()[499]


view_image(pixel[most_accur] ) 
view_image(noise_pixel[most_accur])
view_image(deno_image[most_accur])

view_image(pixel[least_accur])
view_image(noise_pixel[least_accur]) 
view_image(deno_image[least_accur])


#####try different c values#####

def try_c(c, noise_pixel):
     deno_image = np.zeros((500, shape, shape))
     for k in range(500):  
          pi_j = np.ones((shape,shape))/2
          pi_j_prev = np.ones((shape,shape))/2
          for m in range(2,10):
               upper_n1 = np.zeros((shape,shape))
               upper_p1 = np.zeros((shape,shape))
               for i in range(shape):
                    for j in range(shape):
                         if (i in range(1,shape-1) and j in range(1,shape-1)): 
                           upper_p1[i,j] = c*(2*pi_j[i][j+1]-1 + 2*pi_j[i][j-1]-1+ \
                                      (2*pi_j[i-1][j] -1)+ 2* pi_j[i+1][j]-1) + \
                                      2*(noise_pixel[k][i][j+1] + noise_pixel[k][i][j-1] + \
                                         noise_pixel[k][i-1][j] + noise_pixel[k][i+1][j])
                           upper_n1[i,j] = -c*(2*pi_j[i][j+1]-1 + 2*pi_j[i][j-1]-1+ \
                                      (2*pi_j[i-1][j] -1)+ 2* pi_j[i+1][j]-1) -\
                                      2*(noise_pixel[k][i][j+1] + noise_pixel[k][i][j-1] + \
                                         noise_pixel[k][i-1][j] + noise_pixel[k][i+1][j])  
                         
                         elif i == 0 and j == 0:
                               upper_p1[i,j] = c*(2*pi_j[i][j+1]-1 + \
                                      2*pi_j[i+1][j]-1) + \
                                      2*(noise_pixel[k][i][j+1] + \
                                         + noise_pixel[k][i+1][j])
                               upper_n1[i,j] = -c*((2*pi_j[i][j+1]-1) + \
                                      2*pi_j[i+1][j]-1) - \
                                      2*(noise_pixel[k][i][j+1] + \
                                         + noise_pixel[k][i+1][j])
                                      
                         elif i == 0 and j == shape-1:
                              upper_p1[i,j] = c*(2*pi_j[i][j-1]-1 + \
                                      2*pi_j[i+1][j]-1) + \
                                      2*(noise_pixel[k][i][j-1] + \
                                         + noise_pixel[k][i+1][j])
                                      
                              upper_n1[i,j] = -c*(2*pi_j[i][j-1]-1 + \
                                      2*pi_j[i+1][j]-1) - \
                                      2*(noise_pixel[k][i][j-1] + \
                                         + noise_pixel[k][i+1][j])
                                                         
                         elif i == shape-1 and j == 0:
                              upper_p1[i,j] = c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j+1]-1) + \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j+1])
                              upper_n1[i,j] = -c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j+1]-1) - \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j+1])                   
                                      
                         elif i == shape-1 and j == shape-1:
                              upper_p1[i,j] = c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j-1] -1) + \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j-1])
                              upper_n1[i,j] = -c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j-1] -1) - \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j-1])
                         
                         elif j == 0:
                              upper_p1[i,j] = c*(2*pi_j[i-1][j] -1 + \
                                      2*pi_j[i][j+1] -1 +2*pi_j[i+1][j] -1) + \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j+1] +\
                                           noise_pixel[k][i+1][j])
                              upper_n1[i,j] = -c*(2*pi_j[i-1][j] -1 + \
                                      2*pi_j[i][j+1] -1 +2*pi_j[i+1][j] -1) + \
                                      2*(noise_pixel[k][i-1][j]) - \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j+1] +\
                                           noise_pixel[k][i+1][j])
                         elif i == 0:
                              upper_p1[i,j] = c*(2*pi_j[i+1][j]-1 + \
                                      2*pi_j[i][j+1]-1 +2*pi_j[i][j-1]-1 ) + \
                                      2*(noise_pixel[k][i+1][j] + \
                                         + noise_pixel[k][i][j+1] +\
                                           noise_pixel[k][i][j-1])
                              upper_n1[i,j] = -c*(2*pi_j[i+1][j]-1 + \
                                      2*pi_j[i][j+1]-1 +2*pi_j[i][j-1]-1 ) - \
                                      2*(noise_pixel[k][i+1][j] + \
                                         + noise_pixel[k][i][j+1] +\
                                           noise_pixel[k][i][j-1])
                         elif i == shape-1:
                              upper_p1[i,j] = c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j+1] -1 +2*pi_j[i][j-1] -1) + \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j+1] +\
                                           noise_pixel[k][i][j-1])
                              upper_n1[i,j] = -c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j+1]-1 +2*pi_j[i][j-1] -1) - \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i][j+1] +\
                                           noise_pixel[k][i][j-1])
                         else:
                            upper_p1[i,j] = c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j-1] -1 +2*pi_j[i+1][j]-1 ) + \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i+1][j] +\
                                           noise_pixel[k][i][j-1])
                                      
                            upper_n1[i,j] = -c*(2*pi_j[i-1][j]-1 + \
                                      2*pi_j[i][j-1] -1 +2*pi_j[i+1][j]-1 ) - \
                                      2*(noise_pixel[k][i-1][j] + \
                                         + noise_pixel[k][i+1][j] +\
                                           noise_pixel[k][i][j-1])
                  
                         pi_j[i,j] = (np.exp(upper_p1[i,j]) / (np.exp(upper_n1[i,j]) + np.exp(upper_p1[i,j])))
                         
                         if pi_j[i,j] < 0.5:
                              deno_image[k][i][j] = -1
                         else:        
                              deno_image[k][i][j] = 1
                        
     
               Q[k].append(np.sum((pi_j - pi_j_prev) ** 2))
               pi_j_prev = deepcopy(pi_j)  
          
               if Q[k][m-1]<  0.000001:
                    break

     
     return deno_image

deno_image0 = try_c(c = -1, noise_pixel = noise_pixel)
deno_image1 = try_c(c = -0.6, noise_pixel = noise_pixel)
deno_image2 = try_c(c = -0.4, noise_pixel = noise_pixel)
deno_image3 = try_c(c = -0.25, noise_pixel = noise_pixel)
deno_image4 = try_c(c = 0.2, noise_pixel = noise_pixel)
deno_image5 = try_c(c = 0.3, noise_pixel = noise_pixel)
deno_image6 = try_c(c = 0.5, noise_pixel = noise_pixel)
deno_image7 = try_c(c = 0.8 , noise_pixel = noise_pixel)
deno_image8 = try_c(c = 0.9 , noise_pixel = noise_pixel)
deno_image9 = try_c(c = 1, noise_pixel = noise_pixel)

def tpr_fpr(deno_image, pixel = pixel): #get false positive and true positive rate
     fpr = np.zeros(500)
     tpr = np.zeros(500)   
     negative = np.zeros(500)
     positive = np.zeros(500)
     for k in range(500):
          fp = 0
          tp = 0
          negative[k] = np.count_nonzero(pixel[k]==-1)
          positive[k] = shape*shape - negative[k]
          for i in range(shape):
               for j in range(shape): 
                    if deno_image[k][i][j] == 1 and \
                           pixel[k][i][j] == -1:
                           fp += 1
                    fpr[k] = fp / negative[k]
                         
                    if deno_image[k][i][j] == 1 and \
                           pixel[k][i][j] == 1:
                           tp += 1
                    tpr[k] = tp / positive[k]
          
     return np.mean(tpr), np.mean(fpr)

tpr0, fpr0 = tpr_fpr(deno_image0)
tpr1, fpr1 = tpr_fpr(deno_image1)
tpr2, fpr2 = tpr_fpr(deno_image2)
tpr3, fpr3 = tpr_fpr(deno_image3)
tpr4, fpr4 = tpr_fpr(deno_image4)
tpr5, fpr5 = tpr_fpr(deno_image5)
tpr6, fpr6 = tpr_fpr(deno_image6)
tpr7, fpr7 = tpr_fpr(deno_image7)
tpr8, fpr8 = tpr_fpr(deno_image8)
tpr9, fpr9 = tpr_fpr(deno_image9)

tpr_vc = np.array((tpr0, tpr1, tpr2,tpr3, tpr4, tpr5, tpr6, tpr7, tpr8, tpr9))
fpr_vc = np.array((fpr0, fpr1, fpr2, fpr3, fpr4, fpr5, fpr6, fpr7, fpr8, fpr9))

plt.plot(fpr_vc, tpr_vc, "ro")
t = np.arange(0, 1, 0.01)
plt.plot(t, t, "b--")


plt.plot()
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for all c")


