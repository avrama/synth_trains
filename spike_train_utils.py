import numpy as np
import statistics

ms_per_sec=1000
#need to create additional spikes, beyond max_time, to ensure spikes continue to max_time
extra_time=1.1
def summary(spikes,max_time,method=None):
    ISI=[np.diff(x) for x in spikes if len(x)] #replace np.diff with elephant function isi()
    lengths=[len(x) for x in ISI]
    #for i,x in enumerate(ISI):
    #    print(method,'has',len(x),'ISIs for train',i)
    freq=np.mean([np.mean(1/x) for x in ISI if len(x)])
    freq2=np.sum(lengths)/np.shape(lengths)[0]/max_time
    print('###########',method, 'mean number of spikes', np.round(np.mean(lengths)+1),'ff',np.round(freq,3),np.round(freq2,3))
    if np.shape(ISI)[0]<=10:
        print('num ISIs per train:',lengths)
    CV=[np.std(x)/np.mean(x) for x in ISI if len(x)] #replace  with elephant function cv()
    isi_min=np.min([np.min(x) for x in ISI if len(x)])
    isi_max=np.max([np.max(x) for x in ISI if len(x)])
    isi_mean=np.mean([np.mean(x) for x in ISI if len(x)])
    flat_isi=[a for isi in ISI for a in isi]
    isi_med=statistics.median(flat_isi)
    if np.min(lengths)==0:
        print ('      ****',method,'has ', lengths.count(0),'single spike trains, excluded from summary stats')
    if method:
        print('     CV', np.round(np.mean(CV),4),np.round(np.std(CV),4),"ISI min,mean,median,max", np.round(isi_min,5),np.round(isi_mean,5),np.round(isi_med,5),np.round(isi_max,5))
    return ISI,CV,{'min':isi_min,'max':isi_max,'mean':isi_mean, 'median':isi_med, 'CV':np.mean(CV)}

def flatten(isiarray):
    return [item for sublist in isiarray for item in sublist]

def exp_events(mean_isi,Tmax,num_spikes,min_isi):
	isi = np.random.exponential(mean_isi, num_spikes)
	times=np.cumsum(isi[isi>min_isi])
	return times[times< Tmax]  

########################################################
# None of these 1st four methods allow for time dependent variation in firing rate
# All of these methods generate independent trains - see Corr_trains.py on nebish
######################################################
#method 1 from Lytton et al: generate intervals with mean=mean_isi
#Poisson: prob(k; lambda)={lambda^k e^{-lambda}}/{k!}
#lambda = mean and std, k is integer, isi is integer value
#spike time = cumulative sum of intervals
#meanISI is correct, CV too low - very regular spikes - not similar to data?
def spikes_poisson(num_cells,mean_isi,min_isi,max_time):
    num_spikes_per_cell = int(max_time/mean_isi)
    spikesPoisson = []
    for i in range(num_cells):
        isi = np.random.poisson(mean_isi*ms_per_sec, num_spikes_per_cell)/np.float(ms_per_sec)
        spikes=np.cumsum(isi[isi>min_isi])
        spikesPoisson.append(spikes[spikes<max_time])
    ISI,CV_poisson,info=summary(spikesPoisson,max_time,method='poisson')
    return spikesPoisson, info,ISI

#method 2 from Zbyszek: determine how many spike for the cell, then calculate that many spike times between 0 and max_time
#in the loop: eliminate spikes yielding interval too small.  Strange method. 
#Normally distributed with minimum ISI- not similar to data 
#meanISI = max_time/num_spikes_per_cell
def spikes_normal(num_cells,mean_isi,min_isi,max_time):
    num_spikes_per_cell = int(max_time/mean_isi)
    spikesNormal = [sorted(np.random.rand(np.random.poisson(num_spikes_per_cell)) * max_time)
                     for _ in range(num_cells)]
    for i in range(len(spikesNormal)):
        tt = np.array(spikesNormal[i])
        spikesNormal[i] = [tt[0]] + list(tt[1:][np.diff(tt) > min_isi])
    ISI,CV_norm,info=summary(spikesNormal,max_time,method='norm')
    return spikesNormal, info,ISI

#method 3: exp, also known as homogeneous Poisson process.  Could be replaced by elephant function: homogeneous_poisson_process
#spikesExp = [homogeneous_poisson_process(rate=10.0*Hz, t_start=0.0*s, t_stop=100.0*s) for i in range(num_cells)]
def spikes_exp(num_cells,mean_isi,min_isi,max_time):
    num_spikes_per_cell = int(extra_time*max_time/mean_isi)
    spikesExp=[]
    for i in range(num_cells):
        spikesExp.append(exp_events(mean_isi,max_time,num_spikes_per_cell,min_isi))
    ISI,CV_exp,info=summary(spikesExp,max_time,method='exp')
    return spikesExp, info,ISI

#method 4: distribution with mode=intraburst, using mean_isi as mean
#CV, min and max ISI similar to Exp, but slightly longer tail and less truncation at min_isi
#for low firing rates, e.g. SPN, the number of spikes is too low compared to exp
def spikes_lognorm(num_cells,mean_isi,min_isi,max_time,intraburst):
    num_spikes_per_cell = int(extra_time*max_time/mean_isi)
    spikeslogNorm = []
    sigma=np.sqrt((2/3.)*(np.log(mean_isi)-np.log(intraburst)))
    mn=(2*np.log(mean_isi)+np.log(intraburst))/3.
    #sigma=1.5*sigma #This makes number of spikes too low if mean firing rate is quite low
    for i in range(num_cells):
        isi = np.random.randn(num_spikes_per_cell)*sigma+mn
        newisi=np.exp(isi)
        #print('lognorm',sigma,mn,isi,newisi)
        spikes=np.cumsum(newisi[newisi>min_isi])
        spikeslogNorm.append(spikes[spikes<max_time])
    ISI,CV_lognorm,info=summary(spikeslogNorm,max_time,method='lognorm')
    return spikeslogNorm, info,ISI

#method5 from Corr_plot_v2_1.py: use exponential distribution twice, generates a single train
#ISI has high peak and long tailed ISI distribution.  These are exponentially distributed bursts
#good method for bursty data:  
#higher intertrain isi produces lower CV. itisi=0.1 --> 1.0; iti=0.3-->0.7; itisi=0.03-->1.4-1.6
#lower noise = higher CV
def train(cellnum,mean_isi,burstinterval,burstisi,min_isi,max_time,noise):
    num_spikes_per_cell = int(max_time/mean_isi)
    time = 0  
    while time < max_time:
        #t is exponentially distributed time between bursts
        #why not use intertrain isi
        #f(x; 1/beta) = 1/beta *exp(-x/beta),
        t = max(min_isi,np.random.exponential(burstinterval))
        time += t
        #determine how many ISI to generate within the burst - should be spikes_per_burst
        n = max(2,np.random.geometric(1. /num_spikes_per_cell))
        #gaussian distribution of parameter used in exponential, mean = intertrain_isi, sigma=noise
        #noise on the isi provides double dose of noise compared to the spikesExp
        itisi = burstisi + np.random.randn(n) * noise
        #replace any lower than min_isi with min_isi
        itisi=np.array([x if x>min_isi else min_isi for x in itisi])
        #these isis are expoentially distributed within the bursts
        #Issue if itisi=min_isi, which happens if noise too high
        isilist=np.random.exponential(itisi, size=n)
        isis = np.cumsum(isilist[isilist>min_isi])
        times = time + isis
        #print('burst npre',n, 'npost', len(times[times <max_time] ))
        yield times[(times <max_time) & (times >= 0)]
        time = times[-1] if len(times) else time

#Inhomogeneous Poisson process.  Could be replaced by elephant function: inhomogeneous_poisson_process
def spikes_inhomPois(numcells,tdep_rate,time_samp,min_isi,IP_type='InhomPoisson'):
    max_time=time_samp[-1]
    maxrate=np.max(tdep_rate)
    smallest_isi=1/maxrate
    spikes=[]
    for cellnum in range(numcells):
        spike_superset=exp_events(smallest_isi,max_time,int(extra_time*max_time/smallest_isi),min_isi)
        if len(spike_superset):
            spike_rn=np.random.rand(len(spike_superset))
            #find firing rate bin corresponding to spike time
            rate_bin = [np.argmin(np.abs(time_samp-spk)) for spk in spike_superset]
             #normalized spike probability
            prob_spike=tdep_rate[rate_bin]/maxrate
            #retain spikes based on spike probability and random number
            spikes.append(spike_superset[spike_rn<prob_spike])
        else:
            print('uh oh, no spikes generated')
            spikes.append([])
    ISI,CV_IP,info=summary(spikes,max_time,IP_type)
    return spikes, info,ISI

def osc(num_cells,mean_isi,min_isi,max_time,intraburst,interburst,freq_dependence,theta=None,IP_type='osc'):
    samples_per_cycle=10
    if theta:
        maxfreq=max(interburst,theta)
    else:
        maxfreq=1.0/interburst
    freq_sampling_duration=(1.0/maxfreq)/samples_per_cycle
    time_samp=np.arange(0,max_time,freq_sampling_duration)
    #sinusoidal modulation in isi, interburst is 1/sin freq:
    #this gives a mean_freq ~2x 1/mean_isi - not good
    #tdep_rate=1/(mean_isi*(1+freq_dependence*np.sin((2*np.pi/interburst)*time_samp)))
    #sinusoidal modulation in firing rate gives mean number of spikes more similar to exp
    #Also, fft is more unimodal
    tdep_rate=(1./mean_isi)*(1+freq_dependence*np.sin(2*np.pi*time_samp/interburst))
    if theta:
        #This doubles the envelope frequency!
        thetaosc=np.sin(2*np.pi*theta*time_samp)
        #4.5 multiplier increases number of spikes to that for exp and log norm
        #0.2 subtraction produces silent periods between "up" states, probably want these less silent for in vivo
        tdep_rate=(4.5/mean_isi)*((1+freq_dependence*np.sin(2*np.pi*time_samp/interburst))*thetaosc-0.2)
        print('theta',theta,tdep_rate[0:20])
    #
    spikes, info,ISI=spikes_inhomPois(num_cells,tdep_rate,time_samp,min_isi)
    return spikes, info,ISI,time_samp,tdep_rate

def spikes_ramp(num_cells,min_isi,max_time,min_freq,max_freq,start_time,ramp_duration):
    samples_per_cycle=10
    freq_sampling_duration=(1.0/max_freq)/samples_per_cycle
    time_samp=np.arange(0,max_time,freq_sampling_duration)
    tdep_rate=min_freq+max_freq*(time_samp-start_time)*(time_samp>start_time)*(time_samp<(start_time+ramp_duration))
    #
    spikes, info,ISI=spikes_inhomPois(num_cells,tdep_rate,time_samp,min_isi,IP_type='ramp')
    return spikes, info,ISI,time_samp,tdep_rate

def spikes_pulse(num_cells,min_isi,max_time,min_freq,max_freq,start_list,duration):
    samples_per_cycle=10
    freq_sampling_duration=(1.0/max_freq)/samples_per_cycle
    time_samp=np.arange(0,max_time,freq_sampling_duration)
    tdep_rate=min_freq*np.ones(len(time_samp))
    for startt in start_list:
        tdep_rate=tdep_rate+max_freq*(time_samp>startt)*(time_samp<(startt+duration))
    #
    spikes, info,ISI=spikes_inhomPois(num_cells,tdep_rate,time_samp,min_isi,IP_type='pulses')
    return spikes, info,ISI,time_samp,tdep_rate

_FUNCTIONS = {
    #This is not used, because poisson and normal not so good
    'exp': spikes_exp,
    'poisson': spikes_poisson,
    'normal': spikes_normal
}

def make_trains(num_trains,isi,samples,maxTime,train_type):
    print('make trains', train_type)
    if train_type.startswith('lognorm'):
        distr_info=train_type.split()
        print('distribution',distr_info)
        intraburst=float(distr_info[1])
        train_type=distr_info[0]
        spikes,info,ISI=spikes_lognorm(num_trains,isi,samples,maxTime,intraburst)
    else:
        #func=_FUNCTIONS[train_type]
        #spikes,info,ISI=func(num_trains,isi,samples,maxTime)
        spikes,info,ISI=spikes_exp(num_trains,isi,samples,maxTime)
    return spikes,info,ISI

'''
Inhomogeneous Poisson Process from Ujfalussy github R code:
gen.Poisson.events <- function(Tmax, rate){
	## homogeneous Poisson process
	## Tmax: time in ms
	## rate: event frequency in 1/ms
	t0 <- rexp(1, rate)
	if (t0 > Tmax) {
		sp <- NULL
	} else {
		sp <- t0
		tmax <- t0
		while(tmax < Tmax){
			t.next <- tmax + rexp(1, rate)
			if (t.next < Tmax) sp <- c(sp, t.next)
			tmax <- t.next
		}
	}
	return(sp)		
}

NOTE by AB: gen.Poisson.events translated into exp_events, but added min_isi

gen.Poisson.train <- function(rates, N){
	## inhomogeneous Poisson process
	Tmax <- length(rates) # Tmax in ms
	max.rate <- max(rates) # rates in Hz
	t.sp.kept <- c(0, 0) # cell, time (ms)
	
	for (cell in 1:N){	
		t.sp <- gen.Poisson.events(Tmax, max.rate/1000)
		if (length(t.sp > 0)){
			for (tt in t.sp){
				## we keep the transition with some probability
				rr <- rates[ceiling(tt)]
				p.keep <- rr/max.rate # both in Hz!
				if (runif(1) < p.keep){
					t.sp.kept <- rbind(t.sp.kept, c(cell-1, tt))
				}
			}
		}
	}
	t.sp.kept <- t.sp.kept[-1,]
	ii <- sort(t.sp.kept[,2], index.return=T)$ix
	t.sp <- t.sp.kept[ii,]
	t.sp
}
                
NOTE by AB: gen.Poisson.train translated into python above

'''
