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
vowelsAll = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/datasets/vowels.csv', delimiter=' ')
acc = list()
tim = list()

for i in range(30):
    vowels = data (dataset=vowelsAll, percTrain=0.7, percVal=0, percTest=0.3, normType='mean')
    print vowels
    
    #w = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM/weightsFunc.csv', delimiter=' ')
    #hb = np.genfromtxt('/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM/hidBiasFunc.csv', delimiter=' ')
    #hb = np.reshape(hb,(1,hb.shape[0]))
    #weights = np.concatenate ((w, hb), axis = 0)
    
    init = time.time()    
    elmNet = KELM (inTrain=vowels.trainIn, outTrain=vowels.trainOut, kernelType ='pol')
    res, a = elmNet.train_and_test(vowels.testIn,vowels.testOut, aval=True, reg=0.01)    
    end = time.time()    
    acc.append(a)
    tim.append(end-init)
    del(vowels)
    del(elmNet)
    gc.collect()
    
acc = np.asarray(acc)
tim = np.asarray(tim)

print '\nMedia: ', acc.mean(), '\nStd: ', acc.std()
print '\nTime: ', tim.mean(), '\nStd: ', tim.std()

print res











