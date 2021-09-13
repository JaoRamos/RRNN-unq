#########################################
#                                       #
#     An Oscilatory data set generator  #
#            of samples                 #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 2.0 2021    #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from numpy.random import seed
import random as rnd

start_time = time.time()

def generate_trials(size, mem_gap, offset):
    seed(1)
    first_in         = 100 #time to start the first stimulus     
    stim_dur         = 40  #stimulus duration
    stim_noise       = 0.1 #noise
    var_delay_length = 0   #change for a variable length stimulus
    out_gap          = 40  #how much lenth add to the sequence duration    
    sample_size      = size
    rec_noise        = 0
  
    osc_seed_A = np.array([[0],[1]])
    
    osc_y            = np.array([0,1])
    seq_dur          = first_in+2*stim_dur+mem_gap+var_delay_length+out_gap #Sequence duration
    win              = signal.hann(15)
 
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap

    frec           = 12
    out_t          = 120+ stim_dur
    x              = np.arange(seq_dur)
    tipo           = np.sin(np.array(2*np.pi*frec*x))
    #oscilador      = np.sin(2*np.pi*(float(frec)/float(seq_dur))*x)

    trial_types    = np.random.randint(2, size=sample_size)
    x_train_       = np.zeros((sample_size, seq_dur, 1))    
    x_train        = np.zeros((sample_size, seq_dur, 1))    
    
    y_train_       = 0.01* np.ones((sample_size, seq_dur, 1))
    y_train        = 0.01* np.ones((sample_size, seq_dur, 1))
    
    
    for ii in np.arange(sample_size): 
        off = rnd.uniform(-offset, offset) # a ver que hace
        x_train_[ii, first_in:first_in + stim_dur, 0]  = osc_seed_A[trial_types[ii], 0]
        x_train[ii, first_in:first_in + stim_dur, 0]   =signal.convolve(x_train_[ii, first_in:first_in + stim_dur, 0], win + off, mode='same') / sum(win) 
        y_train_[ii, out_t + var_delay[ii]:-out_gap, 0]= osc_y[trial_types[ii]]
        y_train[ii, out_t + var_delay[ii]:-out_gap, 0] =signal.convolve(y_train_[ii, out_t + var_delay[ii]:-out_gap, 0], np.sin(2*np.pi*(float(frec)/float(seq_dur))*x), mode='same')
        #print("Generando... ", ii)


    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]
        
    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    y_train =0.20* y_train + 0*stim_noise * np.random.randn(sample_size, seq_dur, 1)

    a__= y_train[1].T
    a_ =a__[0]
    t_= np.arange(len(y_train[1]))
    #print("a_",a_)
    #print("a_",a__)
    #print( "print t",t_)
    
    peakind_min      = signal.argrelmin(a_,axis=0)
    amp_pico_min_x   = a_[peakind_min]

    print( "peakind min: ",peakind_min)
    print ("Amplitude peak min: ",amp_pico_min_x)#osc_identity

    pepe         =t_[peakind_min]
    print( "pepe",pepe)
    if len(pepe)>4:
        freq         =1/(0.001*float(pepe[-4]-pepe[-5]))
    else:
        freq =0   

    ff='%.4f'%freq
    #print("t",t_)
    print("Time",float(pepe[-1]-pepe[-2])*0.001,"[Sec], Frequency",ff)

    print("--- %s seconds to generate Osc dataset---" % (time.time() - start_time))
    return(x_train, y_train,mask,seq_dur)
   

#To test the rule that you want to teach

sample_size=10

x_train,y_train, mask,seq_dur = generate_trials(sample_size, 200, 0.5) 

fig     = plt.figure(figsize=(6,8))
fig.suptitle("Oscilatory Data Set Training Sample\n (amplitude in arb. units time in mSec)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0],color='g',label="input A")
    plt.plot(y_train[ii, :, 0],color='gray',label="Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    figname = "data_set_osc_sample.png" 
    plt.savefig(figname,dpi=300)
plt.show()



