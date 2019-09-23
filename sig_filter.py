import numpy as np
from scipy.signal import butter, lfilter, freqz

#https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #freq_response(b,a,fs,cutoff)
    y = lfilter(b, a, data)
    return y

def freq_response(b,a,fs,cutoff):
    w, h = freqz(b, a, worN=8000)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
'''
#lowpass filter to see theta frequency
cutoff = 3.0  # desired cutoff frequency of the filter, Hz
rate_lowpass=filt.butter_lowpass_filter(tdep_rate, cutoff, 1/time_samp[1], order=6)
plt.plot(time_samp,rate_lowpass,'k')
'''
