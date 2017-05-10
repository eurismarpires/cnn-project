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
isoletTrain = np.genfromtxt('/home/patcha/Datasets/Isolet/isolet_train.csv', delimiter=',')
isoletTest = np.genfromtxt('/home/patcha/Datasets/Isolet/isolet_test.csv', delimiter=',')


#dnaTrain = np.concatenate((isoletTrain,isoletTest))

acc = list()
tim = list()

for i in range(1):
    isolet = data (train=isoletTrain, test=isoletTest, normType='max')
    print isolet
    
    #w = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM/weightsFunc.csv', delimiter=' ')
    #hb = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM/hidBiasFunc.csv', delimiter=' ')
    #hb = np.reshape(hb,(1,hb.shape[0]))
    #weights = np.concatenate ((w, hb), axis = 0)
        
    init = time.time()    
    elmNet = KELM (inTrain=isolet.trainIn, outTrain=isolet.trainOut, kernelType ='pol')
    res, a = elmNet.train_and_test(isolet.testIn, isolet.testOut, aval=True, reg=1)    
    end = time.time()    
    acc.append(a)
    tim.append(end-init)  
        
    del(isolet)
    del(elmNet)
    gc.collect()
    
    
    
acc = np.asarray(acc)
tim = np.asarray(tim)

print '\nMedia: ', acc.mean(), '\nStd: ', acc.std()
print '\nTime: ', tim.mean(), '\nStd: ', tim.std()











