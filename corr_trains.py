#CorrTrains.py
#Creates Poisson distributed trains.  Either uncorrelated, 
# or with one of two different types of correlation
# Inputs: 1: output filename
#         2: number of trains
#            check_netparams will calculate how many trains you need for your network
#         3: Spike Frequency
#         4: maxTime
#         5: spike train distribution
#         6: type of correlation
#              0: no correlations
#              1: all trains are randomly shifted versions of the first train
#                  shift param is maximum amount shift - additional parameter
#              2: some trains are linear combinations of others
#                  specify correlation value, and program calculates
#                       how many independent trains and
#                       how many dependent trains.
#                    correlation is actually R^2 - so if sqrt(corr_val)=0.5, then half of trains are independent
#         6: for corr type 1 - shift of entire train, for type 2 - corrlation^2, for type 3 - jitter
#
# Future changes:
# incorporate stuff from InputwithCorrelation2.py: within neuron correlation
# AmpaInput2.py: motor and sensory upstates
#
from __future__ import print_function, division
import sys
from matplotlib import pyplot as plt
import numpy as np
import spike_train_utils as stu

min_isi=0.002
ms_per_sec=1000

###############################################################################
#parameters for creating fake spike trains. Replace with argparser
###############################################################################
try:
    args = ARGS.split(",")
    print("ARGS =", ARGS, "commandline=", args)
    do_exit = False
except NameError: #NameError refers to an undefined variable (in this case ARGS)
    args = sys.argv[1:]
    print("commandline =", args)
    do_exit = True

fname=args[0] #set fname=0 to avoid saving data
num_trains=np.int(args[1]) #num cells
Freq=np.float(args[2]) #Mean frequency of the trains
isi=1/Freq
maxTime = float(args[3])
train_distr=args[4] #statistical distribution of single trains

#this parameter controls whether to introduce correlations between trains:
corr=int(args[5])
print('fname=',fname,' num trains',num_trains,' freq', np.round(Freq,1), ' maxTime',np.round(maxTime,1),' distr',train_distr,' corr',corr)
if (corr==1 or corr==3): 
    shift=float(args[6])
    print (shift, type(shift))
if (corr==2):
    corr_val=float(args[6])
    print (corr_val, type(corr_val))

################End of parameter parsing #############
#Notes for future improvements:
#check_connect will calculate number of synapses and trains needed 
#BUT, need to create the neuron prototype first
#from spspine import check_connect, param_net, d1d2
#
#standard_options from moose_nerp provides simulation time (max_time)

############################
#### Uncorrelated trains (if corr==0)
#### Or, correlation will be imparted from fraction_duplicate and check_connect
if (corr==0):
    spikeTime,info,ISI=stu.make_trains(num_trains,isi,min_isi,maxTime,train_distr)

################################
#correlated trains
#
########## method 1 - each train is randomly shifted version ####################
# cross correlogram would be quite high (i.e., 1.0), even for large shifts
if (corr==1):
    spikeTime,info,ISI=stu.make_trains(1,isi,min_isi,maxTime,train_distr)
    for i in range(num_trains-1):
        spikeShift=shift*np.random.uniform()
        #print ("i, shift:", i, spikeShift)
        spikeTime.append(spikeTime[0][:]+spikeShift)

#Second correlation method - linear combination of trains
if (corr==2):
    indep_trains=int(num_trains-np.sqrt(corr_val)*(num_trains-1))
    depend_trains=num_trains-indep_trains
    print ("Dep, Indep, Total:", depend_trains,indep_trains,num_trains)
    #
    #First create the set of independent Trains
    spikeTime,info,ISI=stu.make_trains(indep_trains,isi,min_isi,maxTime,train_distr)
    total_spikes=[len(item) for item in spikeTime]
    spikes_per_train=sum(total_spikes)/indep_trains
    print ("spikes_per_train", spikes_per_train,"Indep SpikeTime shape", np.shape(spikeTime), np.shape(spikeTime[0]))
    #
    #Second, randomly select spikeTimes from independent trains to create dependent Trains
    if (indep_trains<num_trains):
        for i in range(indep_trains,num_trains):
            #1. determine how many spikeTimes to obtain from each indep Train
            samplesPerTrain=np.random.poisson(float(spikes_per_train)/indep_trains,indep_trains)
            spikeTimeTemp=[]
            for j in range(indep_trains):
                #2. from each indep train, select some spikes, eliminating duplicates
                if len(spikeTime[j]):
                    high=len(spikeTime[j])-1
                    indices=list(set(np.sort(np.random.random_integers(0,high,samplesPerTrain[j]))))
                    spikeTimeTemp=np.append(spikeTimeTemp,spikeTime[j][indices])
            print ('spike train %d:' % (i), spikeTimeTemp,'train length', len(spikeTimeTemp))
            #3. after sampling each indepTrain, sort spikes before appending to spikeTime list
            spikeTime.append(np.sort(spikeTimeTemp))
        print ("num trains:", np.shape(spikeTime))

################ method 3 - each train as random shift (jitter) added to each spike
#cross correlogram lower with increased jitter
#shift used as mean and variance
if (corr==3):
    spikeTime,info,ISI=stu.make_trains(1,isi,min_isi,maxTime,train_distr)
    num_spikes=len(spikeTime[0])
    for i in range(num_trains-1):
        jitter=np.random.normal(shift,shift,num_spikes)
        spikeTime.append(spikeTime[0]+jitter)

################Save the spike times array#############
##### Probably could just save to plain text file  ####
if fname !='0':
    np.savez(fname, spikeTime=spikeTime)
else:
    plt.ion()
    plt.figure()
    plt.eventplot(spikeTime)
    if corr>0:
        corval=str(args[6])
    else:
        corval='0'
    plt.suptitle('freq '+str(Freq)+' corr type '+str(corr)+' value '+ corval)

    
