'''
Author: Carlos Alexandre S. Silva
E-mail: carlosalexandress@gmail.com

This class implements the convolution neural network (CNN) with an extreme learning machine (ELM) classifier

'''
import tensorflow as tf
import numpy as np


def Vec2Mat(vet, numLins, numCols):
    # Transform output vector of class id's in a binary matrix
    mat = np.zeros((numLins, numCols))

    for i in range(numLins):
        mat[i, int(vet[i])] = 1.0
 
    return mat


class CNN_ELM_TF(object):
    def __init__(self, sess, imgHSize, imgWSize, paddSize, numClasses, cnnLearningRate, elmInputLen,
                 elmNumNeurons, elmParamC, elmUseBias=True):

        self.sess = sess
        self._init = False

        self.paddSize = paddSize
        self.imgHSize = imgHSize
        self.imgWSize = imgWSize
        self.cnnInputLen = imgHSize*imgWSize
        self.numClasses = numClasses
        
        self.cnnLearningRate = cnnLearningRate
        
        # ELM Vars
        self.elmInputLen = elmInputLen
        self.elmUseBias = elmUseBias
        self.elmInputBiasLen = elmInputLen
        if elmUseBias:
            self.elmInputBiasLen += 1
        self.elmNumNeurons = elmNumNeurons
        self.elmParamC = elmParamC

        self.elmMatW = tf.Variable(tf.random_uniform([self.elmInputBiasLen, self.elmNumNeurons]), trainable=False, dtype=tf.float32)
        self.elmMatU = tf.Variable(tf.zeros([self.elmNumNeurons, self.elmNumNeurons]), trainable=False, dtype=tf.float32)
        self.elmMatV = tf.Variable(tf.zeros([self.elmNumNeurons, self.numClasses]), trainable=False, dtype=tf.float32)
        self.elmMatBeta = tf.Variable(tf.zeros([self.elmNumNeurons, self.numClasses]), trainable=False, dtype=tf.float32)

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, self.cnnInputLen], name="X_Data")
        self.T = tf.placeholder(tf.float32, [None, self.numClasses], name="T_Data")
        self.cnnOutputSize = tf.placeholder(tf.int32, name="cnnOutputSize")

        # Image dimensions after convolutions and poolings
        size_img_fc_H = int((((self.imgHSize + self.paddSize[0] - 5 + 1)/2) - 5 + 1)/2)
        size_img_fc_W = int((((self.imgWSize + self.paddSize[1] - 5 + 1)/2) - 5 + 1)/2)

        # Store layers weight & bias
        self.cnnWeights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.1), trainable=True),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1), trainable=True),
            # fully connected, 7*7*64 inputs, 512 outputs
            'wd1': tf.Variable(tf.random_normal([size_img_fc_H*size_img_fc_W*64, self.elmInputLen], stddev=0.1), trainable=True)
        }
        
        self.cnnBiases = {
            'bc1': tf.Variable(tf.random_normal([32], stddev=0.1), trainable=True),
            'bc2': tf.Variable(tf.random_normal([64], stddev=0.1), trainable=True),
            'bd1': tf.Variable(tf.random_normal([self.elmInputLen], stddev=0.1), trainable=True)
        }
        
        # Updating ELM matrices
        self.mat_IN = tf.placeholder(tf.float32, [None, self.elmInputBiasLen], name="DataIn")
        
        self.mat_H = tf.sigmoid(tf.matmul(self.mat_IN, self.elmMatW))
        new_U = tf.matmul(self.mat_H, self.mat_H, transpose_a=True)
        self.mat_U_assign = self.elmMatU.assign_add(new_U)
        
        new_V = tf.matmul(self.mat_H, self.T, transpose_a=True)
        self.mat_V_assign = self.elmMatV.assign_add(new_V)
        
        new_Beta = tf.matmul(tf.matrix_inverse(tf.add(tf.eye(self.elmNumNeurons)*self.elmParamC, self.elmMatU)), self.elmMatV)
        self.beta_assign = self.elmMatBeta.assign(new_Beta)

        # Restarting ELM matrices
        self.initMatU = self.elmMatU.assign(tf.zeros([self.elmNumNeurons, self.elmNumNeurons]))
        self.initMatV = self.elmMatV.assign(tf.zeros([self.elmNumNeurons, self.numClasses]))
        self.initMatBeta = self.elmMatBeta.assign(tf.zeros([self.elmNumNeurons, self.numClasses]))
        self.initMatW = self.elmMatW.assign((tf.random_uniform([self.elmInputBiasLen, self.elmNumNeurons]) *2)-1)

        self.matW_Temp = tf.placeholder(tf.float32, [self.elmInputBiasLen, self.elmNumNeurons], name="NewMatW")
        self.matW_assign = self.elmMatW.assign(self.matW_Temp)

        self.cost = self.loss()

        self.accur = self.accuracy()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cnnLearningRate).minimize(self.cost)

        self.calcFCCNN = self.convolution()

        self.init()

    def conv2d(self, x_data, W, b, strides=1):
        res = tf.nn.conv2d(x_data, W, strides=[1, strides, strides, 1], padding='VALID')
        res = tf.nn.bias_add(res, b)
        return tf.nn.relu(res)
    
    
    def maxpool2d(self, x_data, k=2):
        return tf.nn.max_pool(x_data, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    def convolution(self):
        # Reshape input pictures
        dataIn = tf.reshape(self.X, shape=[-1, self.imgHSize, self.imgWSize, 1])
        
        # Padding data to keep dimensions divisible in maxpooling process
        padTmp0 = 0
        padTmp1 = 0
        if self.paddSize[0] > 0:
            padTmp0 = int(self.paddSize[0] / 2)
        if self.paddSize[1] > 0:
            padTmp1 = int(self.paddSize[1] / 2)
        if padTmp0 + padTmp1 > 0:
            dataIn = tf.pad(dataIn, paddings=[[0,0], [padTmp0,padTmp0], [padTmp1,padTmp1], [0,0]], mode="CONSTANT")
        
        # Convolution Layer
        conv1 = self.conv2d(dataIn, self.cnnWeights['wc1'], self.cnnBiases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, self.cnnWeights['wc2'], self.cnnBiases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.cnnWeights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.cnnWeights['wd1']), self.cnnBiases['bd1'])

        # Add bias column (CNN fully connected layer = ELM input)
        if self.elmUseBias:
            Bias = tf.ones((self.cnnOutputSize, 1))
            fc1 = tf.concat([fc1, Bias], axis=1, name="ConcatBias")

        return fc1


    def loss(self):
        Fc = self.convolution()
        
        # Calculating ELM output
        outClass = tf.matmul(tf.sigmoid(tf.matmul(Fc, self.elmMatW)), self.elmMatBeta)
        
        lossValue = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outClass, labels=self.T))

        return lossValue
    
    
    def accuracy(self):
        Fc = self.convolution()

        # Calculating ELM output
        outClass = tf.matmul(tf.sigmoid(tf.matmul(Fc, self.elmMatW)), self.elmMatBeta)
            
        lossValue = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outClass, labels=self.T))
        correc = tf.equal(tf.argmax(outClass, 1), tf.argmax(self.T, 1))
        accurValue = tf.reduce_mean(tf.cast(correc, tf.float32))
        
        return lossValue, accurValue, Fc


    def init(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.initMatW)
        self._init = True


    def calcLoss(self, x_data, t_data):
        tMat = t_data
        if t_data.shape[1] == 1:
            tMat = Vec2Mat(t_data, t_data.shape[0], self.numClasses)

        return self.sess.run(self.cost, feed_dict={self.X: x_data,
                                                   self.T: tMat,
                                                   self.cnnOutputSize: x_data.shape[0]})


    def optimCNNWeights(self, x_data, t_data):
        tMat = t_data
        if t_data.shape[1] == 1:
            tMat = Vec2Mat(t_data, t_data.shape[0], self.numClasses)

        self.sess.run(self.optimizer, feed_dict={self.X: x_data,
                                                        self.T: tMat,
                                                        self.cnnOutputSize: x_data.shape[0]})
        

    def calcAccurLoss(self, x_data, t_data):
        tMat = t_data
        if t_data.shape[1] == 1:
            tMat = Vec2Mat(t_data, t_data.shape[0], self.numClasses)

        return self.sess.run(self.accur, feed_dict={self.X: x_data,
                                                    self.T: tMat,
                                                    self.cnnOutputSize: x_data.shape[0]})


    def trainELM(self, fc_cnn, t_data):
        tMat = t_data
        if t_data.shape[1] == 1:
            tMat = Vec2Mat(t_data, t_data.shape[0], self.numClasses)

        self.sess.run(self.mat_U_assign, feed_dict={self.mat_IN: fc_cnn,
                                                    self.cnnOutputSize: fc_cnn.shape[0]})
        self.sess.run(self.mat_V_assign, feed_dict={self.mat_IN: fc_cnn,
                                                    self.T: tMat,
                                                    self.cnnOutputSize: fc_cnn.shape[0]})

        self.sess.run(self.beta_assign)


    def calcFCMatrix(self, x_data, t_data):
        tMat = t_data
        if t_data.shape[1] == 1:
            tMat = Vec2Mat(t_data, t_data.shape[0], self.numClasses)

        HTemp = self.sess.run(self.calcFCCNN, feed_dict={self.X: x_data,
                                                         self.cnnOutputSize: x_data.shape[0]})
        
        self.trainELM(HTemp, tMat)

        return HTemp


    def restartELMModel(self):
        '''
        Restart (zero fill) ELM matrices U, V and Beta
        '''
        self.sess.run(self.initMatU)
        self.sess.run(self.initMatV)
        self.sess.run(self.initMatBeta)


    def restartELMWeights(self, newWeights=None):
        '''
        Restart or assign new value to ELM weights W
        '''
        if newWeights is None:
            self.sess.run(self.initMatW)
        else:
            self.sess.run(self.matW_assign, feed_dict={self.matW_Temp: newWeights})


    def getWeights(self):
        return self.sess.run(self.elmMatW)


    def getBeta(self):
        return self.sess.run(self.elmMatBeta)

    
    def getUV(self):
        return self.sess.run([self.elmMatU, self.elmMatV])
