import wave
import numpy
import matplotlib.pyplot as plt
import numpy as np
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

#https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
#https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

files = ["outside_1.wav", "inside_1.wav"]

for i in range(len(files)):
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

    plt.subplot(4, len(files), i+1)
    plt.plot(time, audio_normalised)
    plt.title('Audio signal')
    plt.ylabel('Amplitude')


    a = autocorrellation(audio_normalised)
    a = a[len(a)//2:]
    plt.subplot(4, len(files), i+2+1)
    plt.plot(time[:len(a)], a)
    plt.title('Autocorelation')
    plt.ylabel('Amplitude')


    windowed_signal = np.hamming(samples) * audio_normalised
    X = power_spectrum_dft(windowed_signal)
    freq_vector = np.fft.fftfreq(samples, d=1/frequency)
    mid = len(X)//2
    plt.subplot(4, len(files), i+4+1)
    plt.plot(freq_vector[:mid], X[:mid])
    plt.title('DFT^2')
    plt.ylabel('Amplitude')

    log_X = np.log(X)
    cepstrum = np.fft.rfft(log_X)
    plt.subplot(4, len(files), i+6+1)
    plt.plot(range(len(cepstrum)), cepstrum)
    plt.title('cepstrum')
    plt.ylabel('Amplitude')

plt.show()

