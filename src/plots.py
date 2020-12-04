import wave
import numpy
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, rfft
from scipy.interpolate import interp1d
from scipy.signal      import argrelextrema
import glob, os, sys
import scipy.signal as sg
from scipy import signal
from scipy.signal import minimum_phase
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

plt.rcParams["figure.figsize"] = (15,15)

def autocorrellation(x):
    result = sg.correlate(x, x, mode='full', method="fft")
    return result

def real_dft(x):
    return np.fft.rfft(x)

def power_spectrum_dft(x):
    return np.abs(np.fft.fft(x))

def idft(x):
    return np.fft.irfft(x)

#https://gist.github.com/anjiro/e148efe17c1e994981638b1a0c6d0954
def eac(sig, winsize=2048):
	"""Return the dominant frequency in a signal."""
	s = np.reshape(sig[:len(sig)//winsize*winsize], (-1, winsize))
	s = np.multiply(s, np.hanning(winsize))
	f = fft(s)
	p = (f.real**2 + f.imag**2)**(1/3)
	f = rfft(p).real
	q = f.sum(0)/s.shape[1]
	q[q < 0] = 0
	intpf = interp1d(np.arange(winsize//2), q[:winsize//2])
	intp = intpf(np.linspace(0, winsize//2-1, winsize))
	qs = q[:winsize//2] - intp[:winsize//2]
	qs[qs < 0] = 0
	return qs

#https://stackoverflow.com/questions/34235530/python-how-to-get-high-and-low-envelope-of-a-signal
def hl_envelopes_idx(s,dmin=3,dmax=4):
    """
    Input :
    s : 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax : int, size of chunks, use this if size of data is too big
    Output :
    lmin,lmax : high/low enveloppe idx of signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    
    """# using the following might help in some case by cutting the signal in "half"
    s_mid = np.mean(s)
    # pre-sorting of locals min based on sign 
    lmin = lmin[s[lmin]<s_mid]
    # pre-sorting of local max based on sign 
    lmax = lmax[s[lmax]>s_mid]"""
    

    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

os.chdir('experiments/audio_at_receiver')
files = ["audio_017.wav", "audio_005.wav", "audio_009.wav"]

for i in range(len(files)):
    #https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
    #https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

    # Read file to get buffer                                                                                               
    ifile = wave.open(files[i])
    samples = 2*ifile.getnframes()
    frequency = ifile.getframerate()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15

    time = [i/frequency for i in range(samples)]
    audio_normalised = audio_as_np_float32 / max_int16
    audio_normalised = audio_normalised / max(audio_normalised)
    sos = signal.butter(30, 100, 'hp', fs=frequency, output='sos')
    audio_normalised = signal.sosfilt(sos, audio_normalised)

    plt.subplot(4, len(files), i+1)
    plt.plot(time, audio_normalised)
    plt.title('Audio signal')
    plt.ylabel('Amplitude')

    a = autocorrellation(audio_normalised)

    a = (a[len(a)//2:])
    """analytic_signal = hilbert(a)
    amplitude_envelope = np.abs(analytic_signal)"""

    #f2 = interp1d(time[:len(a)], amplitude_envelope, kind='cubic')
    #peaks, _ = find_peaks(amplitude_envelope, prominence=1)
    """analytic_signal = hilbert(amplitude_envelope)
    amplitude_envelope = np.abs(analytic_signal)"""

    low_enveloppe, high_enveloppe = hl_envelopes_idx(a)
    low_enveloppe = interp1d(np.array(time)[low_enveloppe], a[low_enveloppe], kind="cubic", fill_value="extrapolate")
    high_enveloppe = interp1d(np.array(time)[high_enveloppe], a[high_enveloppe], kind="cubic", fill_value="extrapolate")

    plt.subplot(4, len(files), i+len(files)+1)
    plt.scatter(time[:len(a)], a, s=1)

    plt.plot(np.array(time), low_enveloppe(time), label='nearest', color='red')
    plt.plot(np.array(time), high_enveloppe(time), label='nearest', color='green')
    #plt.plot(np.array(time), high_enveloppe(time)-low_enveloppe(time), label='nearest', color='black')

    #plt.plot(time[:len(a)], f2(time[:len(a)]), label='envelope', color='red')
    #plt.plot(np.array(time[:len(a)]), amplitude_envelope, color="orange")
    plt.title('Autocorelation')
    plt.ylabel('Amplitude')

    windowed_signal = np.hamming(samples) * audio_normalised
    #windowed_signal = audio_normalised
    X = power_spectrum_dft(windowed_signal)
    freq_vector = np.fft.fftfreq(samples, d=1/frequency)
    mid = len(X)//2
    plt.subplot(4, len(files), i+2*len(files)+1)
    plt.plot(freq_vector[:mid], X[:mid])
    plt.title('DFT^2')
    plt.ylabel('Power')

    cepstrum = eac(windowed_signal)
    plt.subplot(4, len(files), i+3*len(files)+1)
    plt.plot(range(len(cepstrum)), cepstrum)
    plt.title('Enhanced autocorrelation')
    plt.ylabel('Amplitude')

plt.show()

