'''
Created on 14/03/2017

Author: Carlos Alexandre S. Silva
E-mail: carlosalexandress@gmail.com

'''
from __future__ import division
import sys
sys.path.insert (0, '..\\GPU\\') # CNN_ELM, CNN and ELM path
sys.path.insert (0, '..\\..\\RBM-Pacheco\\GPU\\') # RBM path
#import ELM_TF
#import CNN_TF
import CNN_ELM_TF
import rbm_tensorflow
import tensorflow as tf
import numpy as np
import time

if __name__ == '__main__':

    print("Loading Dataset...")
    nomeBase = "vogais"
    if nomeBase == "alfabeto":
        NomeBaseAlfabeto = "C:\\Users\\Carlos\\Desktop\\Doutorado\\Alfabeto\\alfabetoNEW.txt"
        Base = np.matrix(np.loadtxt(NomeBaseAlfabeto, delimiter=','))

        Base = np.hstack((Base[:, -1] - 1, Base[:, :-1]/255))
    
        # Standardization
        #BaseAlfabeto[:, 1:] = (BaseAlfabeto[:, 1:] - numpy.mean(BaseAlfabeto[:, 1:])) / numpy.std(BaseAlfabeto[:, 1:])
    
        size_imgH = 30
        size_imgW = 30
        output_len = 26
        size_padd = [2, 2]

    if nomeBase == "vogais":
        NomeBaseVogais = "C:\\Users\\Carlos\\Desktop\\Doutorado\\Alfabeto\\vowels.csv"
        Base = np.matrix(np.loadtxt(NomeBaseVogais, delimiter=' '))
        # Normalizacao -> Classe esta na ultima coluna, e comeca em 1. Atributos estao entre 0 e 255.
        Base = np.hstack((Base[:, -1] - 1, Base[:, :-1]/255))
    
        size_imgH = 30
        size_imgW = 30
        output_len = 5
        size_padd = [2, 2]

    num_treino = int(Base.shape[0] * 0.7)
    np.random.shuffle(Base)
    BaseTreino = Base[0:num_treino, :]
    BaseTeste = Base[num_treino:, :]
    input_len = size_imgH*size_imgW

    numEXECS = 10
    listaAcur = np.zeros((numEXECS, 1))

    np.set_printoptions(edgeitems=10000, linewidth=10000)

    numEpochsCNN = 30
    learnRateCNN = 0.001

    num_Neurons_ELM = 500


    '''
    # ELM
    print("==== ELM ====")
    for ex in range(numEXECS):
        sess = tf.Session()
        tempototal = time.time()
        hidden_num = 1000
        batch_size = 10000
        
        elm = ELM_TF.ELM(sess, input_len, hidden_num, output_len, paramC=1, useBias=True)
    
        pIni = 0
        pFim = batch_size

        while pIni < BaseTreino.shape[0]:
            elm.calc_mat_U_V(BaseTreino[pIni:pFim, 1:], BaseTreino[pIni:pFim, 0])
            pIni = pFim
            pFim += batch_size
            if pFim > BaseTreino.shape[0]:
                pFim = BaseTreino.shape[0]
    
        # testing
        elm.calc_beta()
        valAcur = elm.test(BaseTeste[:, 1:], BaseTeste[:, 0])

        listaAcur[ex, 0] = valAcur
        
        print("Testing Accuracy: {:.3f}, total time: {} secs".format(valAcur, int(time.time() - tempototal)))
        del(elm)
        sess.close()

    print("\n")
    print("ELM - {:.3f} ({:.3f})\n".format(np.mean(listaAcur[:, 0]), np.std(listaAcur[:, 0]))) 

    #'''


    '''
    # CNN
    print("==== CNN ====")
    for ex in range(numEXECS):
        sess = tf.Session()
        tempototal = time.time()
        learning_rate = learnRateCNN
        nEpochs = numEpochsCNN
        dropOut = 0.5
    
        batchsize = 100
        numBatches = int(BaseTreino.shape[0] / batchsize) # Numero total de batches
        indBatchesExib = int(numBatches / 10) # Numero de batches exibidos
    
        cnn = CNN_TF.CNN(sess, size_imgH, size_imgW, size_padd, output_len, learning_rate)
    
        for epocs in range(nEpochs):
    
            batchIni = 0
            batchFim = batchsize
            while batchIni < BaseTreino.shape[0]:
                
                cnn.otimizaPesos(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0], dropOut)
    
                batchIni = batchFim
                batchFim += batchsize
    
    
        # testing
        _, valAcur = cnn.verificaAcurErro(BaseTeste[:, 1:], BaseTeste[:, 0], 1.0)
        listaAcur[ex, 0] = valAcur
        
        print("Testing Accuracy: {:.3f}, total time: {} secs".format(valAcur, int(time.time() - tempototal)))
        del(cnn)
        sess.close()

    print("\n")
    print("CNN - {:.3f} ({:.3f})\n".format(np.mean(listaAcur[:, 0]), np.std(listaAcur[:, 0]))) 

    #'''

    '''
    # CNN + ELM
    print("==== CNN+ELM ====")
    for ex in range(numEXECS):
        sess = tf.Session()
        
        tempototal = time.time()
        learning_rate = learnRateCNN
        nEpochs = numEpochsCNN
        
        batchsize = 100
    
        num_Input_ELM = 512
        flagBias = True

        cnn_elm = CNN_ELM_TF.CNN_ELM_TF(sess, size_imgH, size_imgW, size_padd, output_len, learning_rate,
                                        elmInputLen=num_Input_ELM, elmNumNeurons=num_Neurons_ELM,
                                        elmParamC=1, elmUseBias=flagBias)
    
        for epocs in range(nEpochs):
    
            np.random.shuffle(BaseTreino)
    
            cnn_elm.restartELMModel()
    
            batchIni = 0
            batchFim = batchsize
            while batchIni < BaseTreino.shape[0]:
                
                if epocs == 0:
                    # Initializing ELM model with CNN features
                    cnn_elm.calcFCMatrix(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0])

                # Optimizing CNN weights (with ELM classifier)
                cnn_elm.optimCNNWeights(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0])
                # Adjusting ELM model with CNN new weigths
                cnn_elm.calcFCMatrix(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0])
    
                batchIni = batchFim
                batchFim += batchsize


        # testing
        _, valAcur, _ = cnn_elm.calcAccurLoss(BaseTeste[:, 1:], BaseTeste[:, 0])
        listaAcur[ex, 0] = valAcur

        print("CNN+ELM - Testing Accuracy: {:.3f}, total time: {} secs".format(valAcur, int(time.time() - tempototal)))
        del(cnn_elm)
        sess.close()

    print("\n")
    print("CNN+ELM - {:.3f} ({:.3f})\n".format(np.mean(listaAcur[:, 0]), np.std(listaAcur[:, 0]))) 

    #'''


    #'''
    # CNN + ELM + RBM
    print("==== CNN+ELM+RBM ====")
    for ex in range(numEXECS):
        sess = tf.Session()
        
        tempototal = time.time()
        learning_rate = learnRateCNN
        nEpochs = numEpochsCNN
        
        batchsize = 100
    
        num_Input_ELM = 1024#512
        flagBias = True

        rbmHidNeurons = num_Neurons_ELM
        maxIterRbm = 50

        
        cnn_elm = CNN_ELM_TF.CNN_ELM_TF(sess, size_imgH, size_imgW, size_padd, output_len, learning_rate,
                                        elmInputLen=num_Input_ELM, elmNumNeurons=num_Neurons_ELM,
                                        elmParamC=1, elmUseBias=flagBias)
    
        for epocs in range(nEpochs):
    
            np.random.shuffle(BaseTreino)
    
            cnn_elm.restartELMModel()
    
            batchIni = 0
            batchFim = batchsize
            while batchIni < BaseTreino.shape[0]:
                
                if epocs == 0:
                    # Initializing ELM model with CNN features
                    Features = cnn_elm.calcFCMatrix(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0])

                if (epocs == 0) or (epocs == nEpochs - 1):
                    # Running RBM with CNN features
                    rbmNet = rbm_tensorflow.RBM_TF(dataIn=Features[:, 1:], numHid=rbmHidNeurons)
                    rbmNet.train(maxIter=maxIterRbm, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=Features.shape[0], freqPrint=10, verbose=False)
                    W = np.concatenate((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)    
                    del(rbmNet)

                    # Optimizing ELM model (W matrix) with RBM output
                    cnn_elm.restartELMWeights(W)

                # Optimizing CNN weights (with ELM classifier)
                cnn_elm.optimCNNWeights(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0])
                # Adjusting ELM model with CNN new weigths
                Features = cnn_elm.calcFCMatrix(BaseTreino[batchIni:batchFim, 1:], BaseTreino[batchIni:batchFim, 0])
    
                batchIni = batchFim
                batchFim += batchsize

        # testing
        _, valAcur, Features = cnn_elm.calcAccurLoss(BaseTeste[:, 1:], BaseTeste[:, 0])
        listaAcur[ex, 0] = valAcur

        print("CNN+ELM+RBM - Testing Accuracy: {:.3f}, total time: {} secs".format(valAcur, int(time.time() - tempototal)))
        del(cnn_elm)
        sess.close()

    print("\n")
    print("CNN+ELM+RBM - {:.3f} ({:.3f})\n".format(np.mean(listaAcur[:, 0]), np.std(listaAcur[:, 0]))) 

    #'''

