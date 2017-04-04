# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This class implements the Extreme Learning Machine (ELM) usinf TensorFlow according to:

[1] Huang, G.B.; Zhu, Q.Y.; Siew, C.-K. Extreme learning machine: theory and applications.
Neurocomputing, v. 70, n. 1, p. 489 - 501, 2006.

Using this class you can either train the net or just execute if you already know the ELM's weight.
All the code is very commented to ease the undertanding.

If you find some bug, please e-mail me =)

'''

import numpy as np
import tensorflow as tf
import sys
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/utils/')
from utils import sigmoid, cont_error

class ELM_TF():
    
    # The constructor method. If you intend to train de ELM, you must fill all parameters.
    # If you already have the weights and wanna only execute the net, just fill W and beta.
    def __init__ (self, neurons=20, inTrain=None, outTrain=None, W=None, beta=None):
        # Setting the neuron's number on the hidden layer        
        self.neurons = neurons      

        
        if inTrain is not None and outTrain is not None:        
            
            # Here we add 1 into the input's matrices to vectorize the bias computation
            self.inTrain = tf.convert_to_tensor (np.concatenate ((inTrain, np.ones([inTrain.shape[0],1])), axis = 1), dtype=tf.float32)
            self.outTrain = tf.convert_to_tensor (outTrain, dtype=tf.float32)
            self.beta = tf.placeholder(tf.float32)
            #self.nSamples = inTrain.shape[0]
          
            # If you wanna initialize the weights W, you just set it up as a parameter. If don't,
            # let the W=None and the code will initialize it here with random values
            if W is not None:
                self.W = tf.convert_to_tensor(W, dtype=tf.float32)
            else:
                # The last row is the hidden layer's bias. Because this we added 1 in the training
                # data above.
                self.W = tf.Variable(tf.random_uniform([inTrain.shape[1]+1,neurons],-1,1,tf.float32))                
                            
                
        else:            
            # In this case, there is no traning. So, you just to fill the weights W and beta
            if beta is not None and W is not None:
                self.beta = tf.convert_to_tensor(beta, dtype=tf.float32)
                self.W = tf.convert_to_tensor(W, dtype=tf.float32)
            else:
                print 'ERROR: you set up the input training as None, but you did no initialize the weights' 
                raise Exception('ELM initialize error')  
                
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        
    
    # This destructor is used just for close de tf.session()
    def __del__(self):
        self.sess.close()        
        
    # This method just trains the ELM. If you wanna check the training error, set aval=True
    def train (self, reg=0.1 , aval=False):
        # Computing the matrix H
        H = tf.sigmoid(tf.matmul(self.inTrain,self.W))        
        
        
        # Computing the weights
        I = tf.eye(self.neurons,dtype=tf.float32)       
        HInv = tf.matmul(tf.matrix_inverse(tf.sub(tf.matmul(H,H,True),reg*I)),H,False,True) 
        self.beta = tf.matmul(HInv,self.outTrain)       
        
        self.sess.run(self.beta)
        
        
        if aval == True:            
            H = tf.sigmoid (tf.matmul(self.inTrain, self.W))
            outNet = tf.matmul(H,self.beta)
            outNetNp = self.sess.run(outNet)
            outTrainNp = self.sess.run(self.outTrain)            
            miss = float(cont_error (outTrainNp, outNetNp))
            si = float(outTrainNp.shape[0])
            print 'Miss classification on the training: ', miss, ' of ', si, ' - Accuracy: ', (1-miss/si)*100, '%'
            
    # This method executes the ELM, according to the weights and the data passed as parameter
    def getResult (self, data, realOutput=None, aval=False):
        # including 1 because the bias
        dataTest = tf.convert_to_tensor(np.concatenate ((data, np.ones([data.shape[0],1])), axis = 1), dtype=tf.float32)            
    
        # Getting the H matrix
        H = tf.sigmoid(tf.matmul(dataTest, self.W))
        netOutput = tf.matmul(H,self.beta)
        netOutputNp = self.sess.run(netOutput)

        if aval:        
            miss = float(cont_error (realOutput, netOutputNp))
            si = float(netOutputNp.shape[0])
            acc = (1-miss/si)*100
            print 'Miss classification on the test: ', miss, ' of ', si, ' - Accuracy: ',acc , '%'       
            return netOutput, acc
            
        return netOutput, None
        
    # This method saves the trained weights as a .csv file
    def saveELM (self, nameFile='T1'):
        np.savetxt('weightW'+nameTrain+'.csv', self.sess.run(self.W))
        np.savetxt('weightBeta'+nameTrain+'.csv', self.sess.run(self.beta))


x = np.zeros([10,5])
y = np.zeros([10,2])

net = ELM_TF (10,x,y)



