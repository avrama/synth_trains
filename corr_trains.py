#CorrTrains.py
#Creates Poisson distributed trains.  Either uncorrelated, 
# or with one of two different types of correlation
# Inputs: 1: output filename - 0 for plots instead of saving file
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
#               3: jitter added to each spike
#         6: for corr type 1 - shift of entire train, for type 2 - corrlation^2, for type 3 - jitter
#         7: number of sets of trains
#
# Future changes:
# incorporate stuff from InputwithCorrelation2.py: within neuron correlation
# AmpaInput2.py: motor and sensory upstates
#Str=num trains:100,freq:4.1,max time: 20.0,distr: exp, corr: 3, jitter:.01,.03,.1,.3, or corr 2, value=0.5,0.64,0.81, 0.9
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
#examples
#python3 corr_trains.py str_exp_corr0.5,100,4.1,20,exp,3 0.03,15
#ARGS="str_exp_corr0.5,100,4.1,20,exp,2 0.5,15"
#exec(open('corr_trains.py').read())

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
#if using lognormal, train_distr should have intraburst ISI separated by space from distr name
#this parameter controls whether to introduce correlations between trains:
#if corr > 0, corr_info should have  corr value separated by space from  corr type
corr_info=args[5]
if corr_info.startswith('0'):
    corr_type=int(corr_info)
else:
    corr_type=int(corr_info.split()[0])
if (corr_type==1 or corr_type==3):
    shift=float(corr_info.split()[1])
    print (shift, type(shift))
if (corr_type==2):
    corr_val=float(corr_info.split()[1])
    print (corr_val, type(corr_val))

if len(args)>6:
    num_sets=int(args[6])
else:
    num_sets=1

print('fname=',fname,' num trains',num_trains,' freq', np.round(Freq,1), ' maxTime',np.round(maxTime,1),' distr',train_distr,' corr',corr_info,'sets',num_sets)
################End of parameter parsing #############
#Notes for future improvements:
#check_connect will calculate number of synapses and trains needed 
#BUT, need to create the neuron prototype first
#from spspine import check_connect, param_net, d1d2
#
#standard_options from moose_nerp provides simulation time (max_time)

for setnum in range(num_sets):
############################
    #### Uncorrelated trains (if corr_type==0)
    #### Or, correlation will be imparted from fraction_duplicate and check_connect
    if (corr_type==0):
        spikeTime,info,ISI=stu.make_trains(num_trains,isi,min_isi,maxTime,train_distr)

    ################################
    #correlated trains
    #
    ########## method 1 - each train is randomly shifted version ####################
    # cross correlogram would be quite high (i.e., 1.0), even for large shifts
    if (corr_type==1):
        spikeTime,info,ISI=stu.make_trains(1,isi,min_isi,maxTime,train_distr)
        for i in range(num_trains-1):
            spikeShift=shift*np.random.uniform()
            #print ("i, shift:", i, spikeShift)
            spikeTime.append(spikeTime[0][:]+spikeShift)

    #Second correlation method - linear combination of trains
    if (corr_type==2):
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
    #
    ################ method 3 - each train as random shift (jitter) added to each spike
    #cross correlogram lower with increased jitter
    #shift used as mean and variance
    if (corr_type==3):
        spikeTime,info,ISI=stu.make_trains(1,isi,min_isi,maxTime,train_distr)
        num_spikes=len(spikeTime[0])
        for i in range(num_trains-1):
            jitter=np.random.normal(shift,shift,num_spikes)
            spike_tmp=spikeTime[0]+jitter
            ########## Prevent negative spiketimes ##############
            good_spike_list=[a and b for a,b in zip(spike_tmp>0,spike_tmp<maxTime)]
            good_spikes=spike_tmp[good_spike_list]
            neg_spikes=maxTime+spike_tmp[spike_tmp<0]
            late_spikes=spike_tmp[spike_tmp>maxTime]-maxTime
            #print('train',i,'fix',spike_tmp[spike_tmp<0],neg_spikes,spike_tmp[spike_tmp>maxTime],late_spikes)
            spikes=np.sort(np.concatenate([good_spikes,neg_spikes,late_spikes]))
            spikeTime.append(spikes)

    ################Save the spike times array#############
    ##### Probably could just save to plain text file  ####
    if fname !='0':
        np.savez(fname+'_t'+str(setnum), spikeTime=spikeTime)
    else:
        plt.ion()
        plt.figure()
        plt.eventplot(spikeTime)
        if corr_type>0:
            corval=str(args[5])
        else:
            corval='0'
        plt.suptitle('freq '+str(Freq)+' corr type '+str(corr_type)+' value '+ corval+' trial '+str(setnum))

    
