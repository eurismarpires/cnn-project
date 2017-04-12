# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

"""

import sys
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/utils/')
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/DataClass')
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM')

import numpy as np
from elm import *
from elm_tensorflow import *
from utils import *
from data import data
from rbm import *
from rbm_tensorflow import *
import time
import gc
import tensorflow as tf

it = 30
hidNeurons = 750
maxIterRbm = 50

# loading the data set
print 'Loading the dataset...'
vowelsAll = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/datasets/vowels.csv', delimiter=' ')

acc = list()
tim = list()
for i in range(it):

    
    
    vowels = data (dataset=vowelsAll, percTrain=0.7, percVal=0, percTest=0.3, normType='mean')
    print vowels
    
    print 'Starting training RBM ', i , ' ...'  
    
    #sess = tf.Session()
    init = time.time()    
    rbmNet = RBM_TF (dataIn=vowels.trainIn, numHid=hidNeurons)
    rbmNet.train (maxIter=maxIterRbm, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=150, freqPrint=10)
    W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)    
    #sess.close()
    print rbmNet.getWeights().shape
    print rbmNet.getHidBias().shape
    print W.shape
    del(rbmNet)
    
    #sess = tf.Session()
    print 'Starting training ELM ', i , ' ...' 
    elmNet = ELM (hidNeurons, vowels.trainIn, vowels.trainOut,W)
    elmNet.train(aval=True)    
    end = time.time()    
    res, a = elmNet.getResult (vowels.testIn,vowels.testOut,True)
    #sess.close()
    
    acc.append(a)
    tim.append(end-init)
    print '\nIteration time: ', end-init, ' sec', 'Predict to end: ', (end-init)*(30-i)/60, ' min'
    
    
    del(vowels)    
    del(elmNet)
    gc.collect()
	
    
acc = np.asarray(acc)
tim = np.asarray(tim)
print '\nMedia: ', acc.mean(), '\nStd: ', acc.std()
print '\nMedia: ', tim.mean(), '\nStd: ', tim.std()











