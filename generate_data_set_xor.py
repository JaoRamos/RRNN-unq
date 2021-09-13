#########################################
#                                       #
#     A "Xor" data set generator        #
#            of samples                 #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 1.0 2021    #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from numpy.random import seed

start_time = time.time()

semillita=2
seed(semillita)

def generate_trials(size, mem_gap):#time loop
    seed(3)
    
    first_in         = 60 #-25#time to start the first stimulus   #30 #60
    stim_dur         = 20 #stimulus duration #20 #30
    stim_noise       = 0.1 #noise level
    var_delay_length = 50 #change for a variable length stimulus
    out_gap          = 140#how much lenth add to the sequence duration 
    sample_size      = size # sample size
    rec_noise        = 0
    
    xor_seed_A       = np.array([[0],[1],[0],[1]])
    xor_seed_B       = np.array([[0],[0],[1],[1]])
    xor_y            = np.array([0,1,1,0])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+out_gap #Sequence duration

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap

    win              = signal.hann(10)

    out_t       = mem_gap+ first_in+stim_dur
    trial_types = np.random.randint(4, size=sample_size)
    x_train     = np.zeros((sample_size, seq_dur, 2))
    x_train_    = np.zeros((sample_size, seq_dur, 2)) 
    y_train     =  np.zeros((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):
        x_train[ii, first_in+var_delay[ii]:first_in + stim_dur+var_delay[ii], 0] = xor_seed_A[trial_types[ii], 0]
        x_train[ii, first_in+var_delay[ii]:first_in + stim_dur+var_delay[ii], 1] = xor_seed_B[trial_types[ii], 0]
      
        y_train[ii, out_t + var_delay[ii]:, 0] = xor_y[trial_types[ii]]

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]
       
    x_train = x_train+stim_noise * np.random.randn(sample_size, seq_dur, 2)# train with or without noisy inputs
    print("--- %s seconds to generate And dataset---" % (time.time() - start_time))
    return (x_train, y_train, mask,seq_dur)

#To see how is the training data set uncoment these lines

sample_size=10
mem_gap=20
x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap) 

#print ("x",x_train)
#print ("y",y_train)

fig     = plt.figure(figsize=(6,8))
fig.suptitle("\"XOR\" Data Set Training Sample\n (amplitude in arb. units time in mS)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0],color='g',label="Input A")
    plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
    plt.plot(y_train[ii, :, 0],color='grey',label="Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
figname = "data_set_and_sample_seed_"+str(semillita)+".png"
plt.savefig(figname,dpi=200)
plt.show()

