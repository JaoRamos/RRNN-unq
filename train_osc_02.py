##########################################################
#                 Author C. Jarne                        #
#            binary_and_recurrent_main.py  (ver 2.0)     #                       
#  An "and" task (low edge triggered)                    #                
#                                                        #
# MIT LICENCE                                            #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import keras
import keras.backend as K
import gc
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, Callback#, warnings
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout, GaussianNoise
from keras import metrics,  activations, initializers, constraints
#from keras import optimizers
from tensorflow.keras import optimizers
from keras import regularizers
#from keras.engine.topology import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import Initializer
from keras.regularizers import l1,l2
import tensorflow as tf

# taking dataset from function
from generate_data_set_oscilator import *

generate_trials
####

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.009, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print(" Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def and_fun(t,N_rec,base,base_plot, offset, epoc):
  
    lista_distancia=[]
    #Parameters

    sample_size      = 15050 #training samples
    epochs           = epoc #training instances
    mem_gap          = t

    # Initializers
    
    pepe  = keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(float(1)/float((N_rec))), seed=None)
    pepe2 =tf.keras.initializers.Orthogonal(gain=1.25, seed=None)

    # Generating data set "the Rules" for training network
    
    x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap, offset)  

    #Network model construction
    seed(None)
    model = Sequential()
    model.add(SimpleRNN(units=N_rec,return_sequences=True,use_bias=True ,input_shape=(None, 1), kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',bias_initializer="zeros" ,activation='tanh'))
    model.add(Dense(units=1,input_dim=N_rec))
    model.save(base+'/00_initial.hdf5')

    ## Model Compiling:
    
    #Optimizers
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    ADAM           = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0001,clipnorm=1.0)#,clipvalue=0.5)#clipnorm=1.0)
    #ADA            = optimizers.Adagrad(learning_rate=0.01)
    
    model.compile(loss = 'mse', optimizer=ADAM, sample_weight_mode="temporal")

    # Saving weigths
    filepath       = base+'/and_weights-{epoch:02d}.hdf5'
    callbacks      = [EarlyStoppingByLossVal(monitor='loss', value=0.00009, verbose=1), ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1)]

    #Network training
    history      = model.fit(x_train[50:sample_size,:,:], y_train[50:sample_size,:,:], epochs=epochs, batch_size=64, callbacks = callbacks,     sample_weight=None,shuffle=True, )

    # Model Testing: 
    x_pred = x_train[0:50,:,:]
    y_pred = model.predict(x_pred)

    print("x_train shape:\n",x_train.shape)
    print("x_pred shape\n",x_pred.shape)
    print("y_train shape\n",y_train.shape)

    fig     = plt.figure(figsize=(6,8))
    fig.suptitle("\"Oscillatory\" Data Set Trainined Output \n (amplitude in arb. units time in mS)",fontsize = 20)
    for ii in np.arange(10):
        plt.subplot(5, 2, ii + 1)
        plt.plot(x_train[ii, :, 0],color='g',label="Input A")
        plt.plot(y_train[ii, :, 0],color='grey',linewidth=3,label="Target output")
        plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
        plt.ylim([-2.5, 2.5])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        a=y_train[ii, :, 0]
        b=y_pred[ii, :, 0]
        a_min_b = np.linalg.norm(a-b)      
        lista_distancia.append(a_min_b)
    figname =  base_plot+"/data_set_sample_trained_osc.png" 
    plt.savefig(figname,dpi=200)
    plt.close()
    #plt.show()

    print(model.summary())

    print ("history keys",(history.history.keys()))

    fig     = plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.title(f'Loss con mem_gap: {mem_gap} y N_rec: {N_rec}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figname = base_plot+"/model_loss.png" 
    plt.savefig(figname,dpi=200)

    #plt.show()
    K.clear_session()
    gc.collect()
    return lista_distancia



