import wave
import numpy
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import scipy.signal as sg

#https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
#https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

# Read file to get buffer                                                                                               
ifile = wave.open(f"single_freq\empty\\1.wav")
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

#Plot autocorrelation
def autocorr(x):
    result = sg.correlate(x, x, mode='full', method="fft")
    return result

a = autocorr(audio_normalised)

plt.plot(range(-len(a)//2, len(a)//2), a)
plt.show()