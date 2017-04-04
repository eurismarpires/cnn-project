# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This class implements the Extreme Learning Machine (ELM) according to:

[1] Huang, G.B.; Zhu, Q.Y.; Siew, C.-K. Extreme learning machine: theory and applications.
Neurocomputing, v. 70, n. 1, p. 489 - 501, 2006.

Using this class you can either train the net or just execute if you already know the ELM's weight.
All the code is very commented to ease the undertanding.

If you find some bug, please e-mail me =)

'''

import numpy as np
import sys
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/utils/')
from utils import sigmoid, cont_error


class ELM:
    neurons = None
    inTrain = None
    outTrain = None
    W = None
    beta = None
    
    # The constructor method. If you intend to train de ELM, you must fill all parameters.
    # If you already have the weights and wanna only execute the net, just fill W and beta.
    def __init__ (self, neurons=20, inTrain=None, outTrain=None, W=None, beta=None):
        # Setting the neuron's number on the hidden layer        
        self.neurons = neurons      

        
        if inTrain is not None and outTrain is not None:        
            
            # Here we add 1 into the input's matrices to vectorize the bias computation
            self.inTrain = np.concatenate ((inTrain, np.ones([inTrain.shape[0],1])), axis = 1)
            self.outTrain = outTrain
          
            # If you wanna initialize the weights W, you just set it up as a parameter. If don't,
            # let the W=None and the code will initialize it here with random values
            if W is not None:
                self.W = W
            else:
                # The last row is the hidden layer's bias. Because this we added 1 in the training
                # data above.
                self.W = np.random.uniform(-1,1,[inTrain.shape[1]+1,neurons])             
        else:            
            # In this case, there is no traning. So, you just to fill the weights W and beta
            if beta is not None and W is not None:
                self.beta = beta
                self.W = W
            else:
                print 'ERROR: you set up the input training as None, but you did no initialize the weights' 
                raise Exception('ELM initialize error')   
            
    # This method just trains the ELM. If you wanna check the training error, set aval=True
    def train (self, aval=False):
        # Computing the matrix H
        H = sigmoid(np.dot(self.inTrain,self.W))
        
        # Computing the weights beta
        self.beta = np.dot(np.linalg.pinv(H),self.outTrain)    
        
        if aval == True:            
            H = sigmoid (np.dot(self.inTrain, self.W))
            outNet = np.dot (H,self.beta)
            miss = float(cont_error (self.outTrain, outNet))
            si = float(self.outTrain.shape[0])
            print 'Miss classification on the training: ', miss, ' of ', si, ' - Accuracy: ', (1-miss/si)*100, '%'
            
            
    # This method executes the ELM, according to the weights and the data passed as parameter
    def getResult (self, data, realOutput=None, aval=False):
        # including 1 because the bias
        dataTest = np.concatenate ((data, np.ones([data.shape[0],1])), axis = 1)        
    
        # Getting the H matrix
        H = sigmoid (np.dot(dataTest, self.W))
        netOutput = np.dot (H,self.beta)

        if aval:        
            miss = float(cont_error (realOutput, netOutput))
            si = float(netOutput.shape[0])
            acc = (1-miss/si)*100
            print 'Miss classification on the test: ', miss, ' of ', si, ' - Accuracy: ',acc , '%'       
            return netOutput, acc
            
        return netOutput, None
        
    # This method saves the trained weights as a .csv file
    def saveELM (self, nameFile='T1'):
        np.savetxt('weightW'+nameTrain+'.csv', self.W)
        np.savetxt('weightBeta'+nameTrain+'.csv', self.beta)
        



     

      







