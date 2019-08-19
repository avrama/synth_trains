# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:25:55 2019

@author: kblackw1
"""
import numpy as np
import spike_train_utils as stu
from matplotlib import pyplot as plt

ms_per_sec=1000
savedata=1 #whether to save data to file or instead plot it
numbins=100 #number of bins for histogram
binwidth=0.002 #adjust so histogram looks good
########################### parameters related to spike train generation
max_time=20#5 #sec
min_isi=0.002

cell_type_dict={}
#isi, interburst and intraburst units are seconds
#interburst: interval between bursts; 1/interburst is frequency used for sinusoidally varying spike trains
#freq_dependence is fraction of mean_isi modulated sinusoidally, [0,1.0), but > 0.9 doesn't work with min_isi=2 ms
#noise used in burst train generation

cell_type_dict['SPN']={'num_cells':100,'mean_isi': 1.0/4.1,'interburst': 4.6,'intraburst': (0.34/(6.9+1)),'noise':0.005,'freq_dependence':0.95}
cell_type_dict['GPe']={'num_cells':35,'mean_isi': 1/29.3,'interburst': 0.5,'intraburst': 0.025,'noise':0.005,'freq_dependence':0.95}
cell_type_dict['STN']={'num_cells':200,'mean_isi': 1/18.,'interburst': 1.5,'intraburst': 0.025,'noise':0.005,'freq_dependence':0.95}
#SPN values:
#mean_isi of 1/2.1 from?
#intratrain isi from KitaFrontSynapticNeurosci5-42
#intraburst: 0.01 is mode from KitaFrontSynapticNeurosci5-42, 0.02 is fastest observed SPN firing frequency, 
#GPe values:
#mean_isi from KitaFrontSynapticNeurosci5-42
#interburst is guestimate; intraburst is mode from KitaFrontSynapticNeurosci5-42
#STN values:
#mean_isi from WilsonNeurosci198-54, mean freq is 18-28
#intraburst: same as GPe until find better estimate
#SPN interburst of 4.6 sec gives ~0.22 Hz, STN interburst of 1.5 --> 0.66 Hz, GPe interburst of 0.5 sec --> 2 Hz

#which method I think is better
best_method={'SPN': 'exp', 'GPe':'lognorm','STN':'lognorm'}
for cell_type,params in cell_type_dict.items():
    info={} #dictionary for storing min,max,mean of ISI and CV
    ISI={}
    hists={}
    print ('##################',cell_type,'#################')
    spikesPoisson, info['poisson'],ISI['poisson']=stu.spikes_poisson(params['num_cells'],params['mean_isi'],min_isi,max_time)
    spikesNormal, info['norm'],ISI['norm']=stu.spikes_normal(params['num_cells'],params['mean_isi'],min_isi,max_time)
    spikesExp, info['exp'],ISI['exp']=stu.spikes_exp(params['num_cells'],params['mean_isi'],min_isi,max_time)
    spikeslogNorm, info['lognorm'],ISI['lognorm']=stu.spikes_lognorm(params['num_cells'],params['mean_isi'],min_isi,max_time,params['intraburst'])
    
    spikesBurst=[sorted(np.hstack(stu.train(cell,params['mean_isi'],float(max_time)/params['num_cells'],params['intraburst'],min_isi,max_time,params['noise'])))
                 for cell in range(params['num_cells'])]
    ISI['burst'],CV_burst,info['burst']=stu.summary(spikesBurst,max_time,'burst')
    
    spikesBurst2=[sorted(np.hstack(stu.train(cell,params['mean_isi'],params['interburst'],params['intraburst'],min_isi,max_time,params['noise'])))
                  for cell in range(params['num_cells'])]
    ISI['burst2'],CV_burst2,info['burst2']=stu.summary(spikesBurst2,max_time,method='burst2')
    
    spikesInhomPois,info['InhomPoisson'],ISI['InhomPoisson'],time_samp,tdep_rate=stu.spikes_inhomPois(params['num_cells'],params['mean_isi'],min_isi,max_time,params['intraburst'],params['interburst'],params['freq_dependence'])
    #
    #### only save the data in spike_sets
    spike_sets={'lognorm':spikeslogNorm,'exp': spikesExp,'burst':spikesBurst,'norm':spikesNormal}
    #,'InhomPoisson':spikesInhomPois,'Burst2': spikesBurst2,'burst':spikesBurst,'norm':spikesNormal,'poisson':spikesPoisson}
    spike_sets={'lognorm':spikeslogNorm,'InhomPoisson':spikesInhomPois,'exp': spikesExp}
    #
    ####################################################################
    ###### Plotting and output
    ####################################################################
    #
    if savedata:
        for method in spike_sets.keys():
            fname=cell_type+'_'+method+'_freq'+str(np.round(1/params['mean_isi']))+'.npz'
            np.savez(fname, spikeTime=spike_sets[method], info=info[method])
    else:
        ################# histogram of ISIs ########################3
        min_max=[np.min([info['norm']['min'],info['exp']['min'],info['lognorm']['min'],info['poisson']['min']]),
                 np.max([info['norm']['max'],info['exp']['max'],info['lognorm']['max'],info['poisson']['max']])]
        bins=10 ** np.linspace(np.log10(min_max[0]), np.log10(min_max[1]), numbins)
        bins_IP=list(time_samp)+[max_time]
        hist_dt=np.diff(time_samp)[0]
        for method in spike_sets.keys():
            hists[method],tmp=np.histogram(stu.flatten(ISI[method]),bins=bins,range=min_max)
            #recalculate histogram for inhomogeneous Poisson
        hist_IP,tmp=np.histogram(stu.flatten(spikesInhomPois),bins=bins_IP)
        #
        ########## plot Inhomogeneous Poisson, and also fft
        plot_bins=[(bins_IP[i]+bins_IP[i+1])/2 for i in range(len(bins_IP)-1)]
        plt.ion()
        plt.figure()
        plt.title(cell_type+' time histogram of inhomogenous Poisson')
        plt.bar(plot_bins,hist_IP,width=hist_dt)
        plt.plot(time_samp,np.max(hist_IP)*tdep_rate/np.max(tdep_rate),'r')
        plt.xlabel('time')
        plt.ylabel('num spikes')
        plt.figure()
        plt.title(cell_type+' fft of inhomogenous Poisson')
        fft_IP=np.fft.rfft(hist_IP)
        xf = np.linspace(0.0, 1.0/(2.0*bins_IP[1]), len(fft_IP))
        plt.plot(xf[1:],2/len(fft_IP)*np.abs(fft_IP[1:]))
        ######### plot raster and histogram for other spike trains   
        colors=plt.get_cmap('viridis')
        #colors=plt.get_cmap('gist_heat')
        color_num=[int(cellnum*(colors.N/params['num_cells'])) for cellnum in range(params['num_cells'])]
        color_set=np.array([colors.__call__(color) for color in color_num])
        for labl,spikes in spike_sets.items():
            plt.figure()
            plt.title(cell_type+' '+labl+' raster'+' mean '+str(np.round(info[labl]['mean'],3))+', median '+str(np.round(info[labl]['median'],3)))
            #for i in range(params['num_cells']):
            plt.eventplot(spikes,color=color_set)#[i],lineoffsets=i)
            plt.xlim([0,max_time])
            plt.xlabel('time')
            plt.ylabel('neuron')
        #
        #plt.figure()
        #plt.title('histogram')
        color_num=[int(histnum*(colors.N/len(hists))) for histnum in range(len(hists))]
        plot_bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        flat_hist=stu.flatten([hists[labl] for labl in spike_sets.keys()])
        ymax=np.max(flat_hist)
        ymax=np.mean(flat_hist)+1*np.std(flat_hist)
        xmax=5*np.max([info[t]['median'] for t in spike_sets.keys()])
        plt.figure()
        plt.title(cell_type+' '+' histogram')
        for i,labl in enumerate(spike_sets.keys()):
            plt.bar(np.array(plot_bins)+binwidth*0.1*i,hists[labl], label=labl+' '+str(np.round(1/info[labl]['mean'],1))+' hz '+str(np.round(1/info[labl]['median'],1))+' hz',color=colors.__call__(color_num[i]),width=binwidth)
            plt.xlim([0,xmax])
        #plt.ylim([0,ymax])
        plt.xlabel('ISI')
        plt.ylabel('num events')
        #plt.xticks(plot_bins)
        #plt.xscale('log')
        plt.legend()

