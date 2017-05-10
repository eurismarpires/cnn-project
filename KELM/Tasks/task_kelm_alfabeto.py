# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

"""

import sys
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/utils/')
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/DataClass')
#sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM')
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/KELM')

import numpy as np
#from elm import *
#from elm_tensorflow import *
from kelm import *
from utils import *
from data import data
#from rbm import *
import time
import gc


# loading the data set
print 'Loading the dataset...'
alfabetoAll = np.genfromtxt('/home/patcha/Datasets/Alfabeto/alfabeto.csv', delimiter=',')
acc = list()
tim = list()

for i in range(10):
    alfabeto = data (dataset=alfabetoAll, percTrain=0.7, percVal=0, percTest=0.3, normType='mean')
    print alfabeto
    
    #w = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM/weightsFunc.csv', delimiter=' ')
    #hb = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM/hidBiasFunc.csv', delimiter=' ')
    #hb = np.reshape(hb,(1,hb.shape[0]))
    #weights = np.concatenate ((w, hb), axis = 0)
    
    init = time.time()    
    elmNet = KELM (inTrain=alfabeto.trainIn, outTrain=alfabeto.trainOut, kernelType ='pol')
    res, a = elmNet.train_and_test(alfabeto.testIn,alfabeto.testOut, aval=True, reg=0.001, deg=2)     
    end = time.time()    
    acc.append(a)
    tim.append(end-init)  
    
    del(alfabeto)
    del(elmNet)
    gc.collect()
    
    
    
acc = np.asarray(acc)
tim = np.asarray(tim)

print '\nMedia: ', acc.mean(), '\nStd: ', acc.std()
print '\nTime: ', tim.mean(), '\nStd: ', tim.std()











