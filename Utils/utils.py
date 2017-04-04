# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This file contains the utils function that help to compute the others algorithm

If you find some bug, please, e-mail me

'''

import numpy as np

# This function binarizes a vector
# Example:
# In: v = [1,2,3]
# Out: v = [1,0,0;
#                 0,1,0;
#                 0,0,1]
def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)


# This function sets 1 to the the maximum label and 0 for the other ones
# The input is a matrix of lables, one per row
# Example:
#         [0.1 0.3  0.9
# In:     0.3 0.01 0.2
#          0.9 0.8  0.1]
#
#         [0 0 1
# Out:  1 0 0
#          1 0 0]
def get_max_label (vin):
      [m,n] = vin.shape
      vout = np.zeros([m,n])

      mx = vin.max(axis=1)
      for i in range(m):
            for j in range (n):
                  vout[i,j] = int (mx[i] == vin[i,j])

      return vout


# This function counts the number of the miss classification by comparing the label matrix obtaineg by a classifier
# and the real label matrix. Important: the vreal and vclass must be in this order!
def cont_error (vreal, vclass):
      # Getting the matrix binarized
      vclass = get_max_label (vclass)
      [m,n] = vreal.shape
      dif =vreal - vclass
      err = 0

      for i in range(m):
            flag = 0
            for j in range (n):
                  if dif[i,j] != 0:
                        flag = 1

            if flag == 1:
                  err = err + 1

      return err

# This function computes the sigmoid
def sigmoid (v):
    return 1/(1+np.exp(-v))

# This function computes the RMSE of set of data organized in a matrix. Each line is a sample
# and each column is an attribute. Important: if the np.array is in the format (n,), 
# it must be reshaped to (1,n)
def rmse (x,y):
    return np.sum(np.sqrt(np.sum(np.power(x-y,2),axis=1)/x.shape[1]))/x.shape[0]
    
# This function normalizes the data with zero mean and standart deviation 1
def normZeroMean (data):
    return (data - data.mean())/data.std()
    

# this function shuffles the data respecting its labels values
def shuffleData (dataIn, dataOut):
    n1 = len(dataIn)
    n2 = len(dataOut)
    if n1 != n2:
        raise ('ERROR: the length of dataIn and dataOut must be equal')
    
    pos = np.random.permutation(n1)    
    newIn = list()
    newOut = list()
   
    for i in range(n1):
        newIn.append (dataIn[pos[i]])        
        newOut.append(dataOut[pos[i]])

    return newIn, newOut

# This function flats a list of matrices
def flatList (l):
    n = len(l)
    ret = list()
    for i in range(n):
        ret.append (l[i].flatten())
    return ret
    
# This function split the dataIn and dataOut to dataIn_train, dataOut_train, 
# dataIn_test and dataOut_test. The % of the train and test set is determined
# by pctTrain. Ex: If pctTrain = 0.7 => 70% for training and 30% for test.
# t is the output type: linear or binary
def splitTrainTest (dataIn, dataOut, pctTrain,t = 'linear'):
    if pctTrain > 1 or pctTrain < 0:
        raise ('ERROR: the pctTrain must be in the [0,1] interval')
    
    nsp = len(dataIn) # number of samples
    sli = int(round(nsp*pctTrain)) # getting pctTrain% to trainning   
    dataIn_train = dataIn[0:sli]
    dataIn_test = dataIn[sli:nsp]    
    if t == 'linear':
        dataOut_train = (dataOut[0:sli])
        dataOut_test = (dataOut[sli:nsp])
    elif t == 'binary':
        dataOut_train = ind2vec(dataOut[0:sli])
        dataOut_test = ind2vec(dataOut[sli:nsp])
    
    return dataIn_train, dataOut_train, dataIn_test, dataOut_test
    

# This function returns the confusion matrix
def confusionMatrix (real, net):
    net = get_max_label (net)
    cNet = 0
    cReal = 0
    m,n = real.shape
    mat = np.zeros_like (real)
    
    for j in range(n):
        for i in range(m):
            if real[i,j] == 1:
                cReal = i
            if net[i,j] == 1:
                cNet = i
            mat[cReal,cNet] = mat[cReal,cNet] + 1
            
    return mat
    
# This funtion remount an grayscale image, flatted in an array, into an image format.
# Data is the image flatted and res is the real image's resolution. In this case, the img
# must be square, e.g, res x res
# Ex: let p(i,j) a pixel, an image would be:
# Img = [p(1,1) ... p(1,res)
#          .           .
#          .           .
#          .           .
#       p(res,1) ... p(1,res)]
#
# data = [p(1,1), p(1,2)...p(res,1)...p(res,res)]    
def remountImg (data, res):    
    newImg = np.zeros([res,res])        
    for i in range(res):
        for j in range(res):
            newImg[i,j] = data[i*res + j]
    
    return newImg
    
    
    
    
    
