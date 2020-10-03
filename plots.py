import wave
import numpy
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, rfft
from scipy.interpolate import interp1d
from scipy.signal      import argrelextrema
import glob, os
import scipy.signal as sg
from scipy import signal
from scipy.signal import minimum_phase
plt.rcParams["figure.figsize"] = (15,15)

def autocorrellation(x):
    result = sg.correlate(x, x, mode='full', method="fft")
    return result

def real_dft(x):
    return np.fft.rfft(x)

def power_spectrum_dft(x):
    return np.power(np.abs(np.fft.fft(x)), 2)

def idft(x):
    return np.fft.irfft(x)

#https://gist.github.com/anjiro/e148efe17c1e994981638b1a0c6d0954
def eac(sig, winsize=1024):
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

files = ["outside_long.wav", "inside_long.wav"]

for i in range(len(files)):
    #https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
    #https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

    # Read file to get buffer                                                                                               
    ifile = wave.open(files[i])
    samples = ifile.getnframes()
    frequency = ifile.getframerate()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15

    time = [i/frequency for i in range(samples)]
    audio_normalised = audio_as_np_float32 / max_int16

    sos = signal.butter(30, 100, 'hp', fs=frequency, output='sos')
    audio_normalised = signal.sosfilt(sos, audio_normalised)

    plt.subplot(4, len(files), i+1)
    plt.plot(time, audio_normalised)
    plt.title('Audio signal')
    plt.ylabel('Amplitude')


    a = autocorrellation(audio_normalised)
    a = a[len(a)//2:]
    plt.subplot(4, len(files), i+2+1)
    plt.scatter(time[:len(a)], a, s=1)
    plt.title('Autocorelation')
    plt.ylabel('Amplitude')

    windowed_signal = np.hamming(samples) * audio_normalised
    #windowed_signal = audio_normalised
    X = power_spectrum_dft(windowed_signal)
    freq_vector = np.fft.fftfreq(samples, d=1/frequency)
    mid = len(X)//2
    plt.subplot(4, len(files), i+4+1)
    plt.plot(freq_vector[:mid], X[:mid])
    plt.title('DFT^2')
    plt.ylabel('Power')

    cepstrum = eac(windowed_signal)
    plt.subplot(4, len(files), i+6+1)
    plt.plot(range(len(cepstrum)), cepstrum)
    plt.title('Enhanced autocorrelation')
    plt.ylabel('Amplitude')

plt.show()

