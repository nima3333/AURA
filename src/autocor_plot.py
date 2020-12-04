import wave
import numpy
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import scipy.signal as sg
from scipy import signal
from scipy.signal import minimum_phase

#https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
#https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

# Read file to get buffer                                                                                               
ifile = wave.open(f"outside_1.wav")
samples = ifile.getnframes()
frequency = ifile.getframerate()
audio = ifile.readframes(samples)

# Convert buffer to float32 using NumPy                                                                                 
audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)

# Normalise float32 array so that values are between -1.0 and +1.0                                                      
max_int16 = 2**15
audio_normalised = audio_as_np_float32 / max_int16

time = [i/frequency for i in range(samples)]
#Plot audio signal
"""plt.plot(time, audio_normalised)
plt.show()"""

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


#Plot autocorrelation
def autocorr(x):
    x = butter_highpass_filter(x,10,1000)
    result = sg.correlate(x, x, mode='full', method="fft")
    return result

a = autocorr(audio_normalised)

"""plt.plot(range(0, len(a)//2+1), a[len(a)//2:]/a[len(a)//2])
plt.show()"""

ceps = minimum_phase(audio_normalised)
plt.plot(range(len(ceps)), ceps)
plt.show()