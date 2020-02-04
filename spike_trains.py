# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:25:55 2019

@author: kblackw1
"""
import numpy as np
import spike_train_utils as stu
import sig_filter as filt
from matplotlib import pyplot as plt

ms_per_sec=1000
numbins=100 #number of bins for histogram
binwidth=0.002 #adjust so histogram looks good
########################### parameters related to spike train generation
max_time=5 #sec #20
min_isi=0.002
#these should go in dictionary?
ramp_start=2
ramp_duration=0.3 #fast ~0.3, slow ~0.5
min_freq=3
max_freq=16
pulse_start=[2,2.2]
pulse_duration=0.05

save_spikes={'Ctx': ['exp','ramp', 'osc'], 'STN': ['pulse']}
#save_spikes={'Ctx': ['ramp']}
#save_spikes={} #if empty, will not save data
cell_type_dict={}
#isi, interburst and intraburst units are seconds
#interburst: interval between bursts; 1/interburst is frequency used for sinusoidally varying spike trains
#freq_dependence is fraction of mean_isi modulated sinusoidally, [0,1.0), but > 0.9 doesn't work with min_isi=2 ms
#noise used in burst train generation

#cell_type_dict['str']={'num_cells':100,'mean_isi': 1.0/4.1,'interburst': 1/0.2,'intraburst': 0.19,'noise':0.005,'freq_dependence':0.95}
#cell_type_dict['GPe']={'num_cells':35,'mean_isi': 1/29.3,'interburst': 0.5,'intraburst': 0.027,'noise':0.005,'freq_dependence':0.95}
cell_type_dict['STN']={'num_cells':2000,'mean_isi': 1/18.,'interburst': 1.5,'intraburst': 0.044,'noise':0.005,'freq_dependence':0.95}
cell_type_dict['Ctx']={'num_cells':10000,'mean_isi': 1/10.,'interburst': 1.5,'intraburst': 0.044,'noise':0.005,'freq_dependence':0.95}
#using intraburst of 0.015 gives mean isi too small and mean freq too high!
#cell_type_dict['GPe']={'num_cells':35,'mean_isi': 1/29.3,'interburst': 1/20.,'intraburst': 0.015,'noise':0.005,'freq_dependence':0.5}
#cell_type_dict['STN']={'num_cells':200,'mean_isi': 1/18.,'interburst': 1/20.,'intraburst': 0.015,'noise':0.005,'freq_dependence':0.5}
########### Burst gives different number of spikes.  What is relation among mean isi,interburst and intraburst
#to produce similar number of spikes for burst, need intraburst=0.8/freq 
#theta frequencies for creating doubly oscillatory trains
DMfreq=10.5 #carrier,interburst=0.555Hz
DLfreq=5.0 #carrier,interburst=0.2Hz
thetafreq=0#DLfreq
#str values:
#mean_isi of 1/2.1 from?
#intratrain isi from KitaFrontSynapticNeurosci5-42
#intraburst: 0.01 is mode from KitaFrontSynapticNeurosci5-42, 0.02 is fastest observed str firing frequency, 
#GPe values:
#mean_isi from KitaFrontSynapticNeurosci5-42
#interburst is guestimate; intraburst is mode from KitaFrontSynapticNeurosci5-42
#STN values:
#mean_isi from WilsonNeurosci198-54, mean freq is 18-28
#intraburst: same as GPe until find better estimate
#str interburst of 4.6 sec gives ~0.22 Hz, STN interburst of 1.5 --> 0.66 Hz, GPe interburst of 0.5 sec --> 2 Hz

#which method I think is better

best_method={'str': 'exp', 'GPe':'lognorm','STN':'lognorm','Ctx':'exp'}
for cell_type,params in cell_type_dict.items():
    info={} #dictionary for storing min,max,mean of ISI and CV
    ISI={}; hists={}
    spikes={}
    time_samp={};tdep_rate={} #for Inhomogenous Poisson methods
    print ('##################',cell_type,'#################')
    #spikesPoisson, info['poisson'],ISI['poisson']=stu.spikes_poisson(params['num_cells'],params['mean_isi'],min_isi,max_time)
    #spikesNormal, info['norm'],ISI['norm']=stu.spikes_normal(params['num_cells'],params['mean_isi'],min_isi,max_time)
    spikes['exp'], info['exp'],ISI['exp']=stu.spikes_exp(params['num_cells'],params['mean_isi'],min_isi,max_time)
    spikes['lognorm'], info['lognorm'],ISI['lognorm']=stu.spikes_lognorm(params['num_cells'],params['mean_isi'],min_isi,max_time,params['intraburst'])
    
    '''spikes['Burst']=[sorted(np.hstack(stu.train(cell,params['mean_isi'],float(max_time)/params['num_cells'],params['intraburst'],min_isi,max_time,params['noise'])))
                 for cell in range(params['num_cells'])]
    ISI['burst'],CV_burst,info['burst']=stu.summary(spikesBurst,max_time,'burst')
    
    spikes['Burst2']=[sorted(np.hstack(stu.train(cell,params['mean_isi'],params['interburst'],params['intraburst'],min_isi,max_time,params['noise'])))
                  for cell in range(params['num_cells'])]
    ISI['burst2'],CV_burst2,info['burst2']=stu.summary(spikesBurst2,max_time,method='burst2')
    '''
    spikes['osc'],info['osc'],ISI['osc'],time_samp['osc'],tdep_rate['osc']=stu.osc(params['num_cells'],params['mean_isi'],min_isi,max_time,params['intraburst'],params['interburst'],params['freq_dependence'])
    spikes['ramp'],info['ramp'],ISI['ramp'],time_samp['ramp'],tdep_rate['ramp']=stu.spikes_ramp(params['num_cells'],min_isi,max_time,min_freq,max_freq,ramp_start,ramp_duration)
    spikes['pulse'],info['pulse'],ISI['pulse'],time_samp['pulse'],tdep_rate['pulse']=stu.spikes_pulse(params['num_cells'],min_isi,max_time,min_freq,max_freq,pulse_start,pulse_duration)
    #
    ####################################################################
    ###### Plotting and output
    ####################################################################
    #
    if len(save_spikes):
        for method in save_spikes[cell_type]:
            fname=cell_type+str(cell_type_dict[cell_type]['num_cells'])+'_'+method+'_freq'+str(np.round(1/params['mean_isi']))
            if method=='osc':
                fname=fname+'_osc'+str(np.round(1.0/params['interburst'],1))
                if thetafreq:
                    fname=fname+'_theta'+str(np.round(thetafreq))
            if method=='ramp':
                fname=cell_type+str(cell_type_dict[cell_type]['num_cells'])+'_'+method+str(ramp_duration)+'_freq'+str(min_freq)+'_'+str(max_freq)
            if method=='pulse':
                fname=cell_type+str(cell_type_dict[cell_type]['num_cells'])+'_'+method+'_freq'+str(min_freq)+'_'+str(max_freq)
            print('saving data to', fname)
            np.savez(fname+'.npz', spikeTime=spikes[method], info=info[method])
    else:
        ################# histogram of ISIs ########################3
        min_max=[np.min([info[key]['min'] for key in info.keys()]),
                 np.max([info[key]['max'] for key in info.keys()])]
        bins=10 ** np.linspace(np.log10(min_max[0]), np.log10(min_max[1]), numbins)
        bins_IP={};hist_dt={};time_hist={}
        for ip_type in tdep_rate.keys():
            bins_IP[ip_type]=list(time_samp[ip_type])+[max_time]
            hist_dt[ip_type]=np.diff(time_samp[ip_type])[0]
        for method in spikes.keys():
            hists[method],tmp=np.histogram(stu.flatten(ISI[method]),bins=bins,range=min_max)
            #recalculate histogram for inhomogeneous Poisson
        for ip_type in tdep_rate.keys():
            time_hist[ip_type],tmp=np.histogram(stu.flatten(spikes[ip_type]),bins=bins_IP[ip_type])
        #
        ########## plot Inhomogeneous Poisson, and also fft
        ###### Extract low frequency envelope of signal, only if theta
        for ip_type in tdep_rate.keys():
            plot_bins=[(bins_IP[ip_type][i]+bins_IP[ip_type][i+1])/2 for i in range(len(bins_IP[ip_type])-1)]
            plt.ion()
            plt.figure()
            plt.title(cell_type+' time histogram of '+ ip_type)
            plt.bar(plot_bins,time_hist[ip_type],width=hist_dt[ip_type],label='hist')
            plt.plot(time_samp[ip_type],np.max(time_hist[ip_type])*tdep_rate[ip_type]/np.max(tdep_rate[ip_type]),'r',label='tdep_rate')
            if thetafreq and ip_type=='osc':
                data=time_hist['osc']#tdep_rate
                meandata=np.mean(data)
                newdata=np.abs(data-meandata)
                fft_env=np.fft.rfft(newdata)
                cutoff=3
                fft_lowpas=filt.butter_lowpass_filter(fft_env, cutoff, 1/time_samp[1], order=6)
                plt.plot(time_samp[ip_type],newdata,'k',label='norm hist')
            plt.xlabel('time (sec)')
            plt.ylabel('num spikes')
            plt.legend()
            if ip_type=='osc':
                #plot FFT of histogram
                plt.figure()
                plt.title(cell_type+' fft of time histogram of IP: '+ip_type)
                fft_IP=np.fft.rfft(time_hist[ip_type])
                xf = np.linspace(0.0, 1.0/(2.0*bins_IP[ip_type][1]), len(fft_IP))
                plt.plot(xf[1:],2/len(fft_IP)*np.abs(fft_IP[1:]),label='fft')
                if thetafreq:
                    plt.plot(xf[1:],2/len(fft_lowpas)*np.abs(fft_lowpas[1:]),label='lowpass')
                plt.legend()
        ######### plot raster and histogram for other spike trains   
        colors=plt.get_cmap('viridis')
        #colors=plt.get_cmap('gist_heat')
        color_num=[int(cellnum*(colors.N/params['num_cells'])) for cellnum in range(params['num_cells'])]
        color_set=np.array([colors.__call__(color) for color in color_num])
        for labl,spike_set in spikes.items():
            plt.figure()
            plt.title(cell_type+' '+labl+' raster'+' mean '+str(np.round(info[labl]['mean'],3))+', median '+str(np.round(info[labl]['median'],3)))
            #for i in range(params['num_cells']):
            plt.eventplot(spike_set,color=color_set)#[i],lineoffsets=i)
            plt.xlim([0,max_time])
            plt.xlabel('Time (sec)')
            plt.ylabel('neuron')
        #
        #plt.figure()
        #plt.title('histogram')
        color_num=[int(histnum*(colors.N/len(hists))) for histnum in range(len(hists))]
        plot_bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        flat_hist=stu.flatten([hists[labl] for labl in spikes.keys()])
        ymax=np.max(flat_hist)
        ymax=np.mean(flat_hist)+1*np.std(flat_hist)
        xmax=5*np.max([info[t]['median'] for t in spikes.keys()])
        plt.figure()
        plt.title(cell_type+' '+' histogram')
        for i,labl in enumerate(spikes.keys()):
            plt.bar(np.array(plot_bins)+binwidth*0.1*i,hists[labl], label=labl+' '+str(np.round(1/info[labl]['mean'],1))+' hz '+str(np.round(1/info[labl]['median'],1))+' hz',color=colors.__call__(color_num[i]),width=binwidth)
            plt.xlim([0,xmax])
        #plt.ylim([0,ymax])
        plt.xlabel('ISI')
        plt.ylabel('num events')
        #plt.xticks(plot_bins)
        #plt.xscale('log')
        plt.legend()

