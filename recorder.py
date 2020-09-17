#https://realpython.com/playing-and-recording-sound-python/#recording-audio
#https://gist.github.com/akey7/94ff0b4a4caf70b98f0135c1cd79aff3

#Do not work

import numpy as np
import sounddevice as sd
import time
from scipy.io.wavfile import write
from threading import Thread

def record():
    fs = 44100
    record_duration = 10
    myrecording = sd.rec(int(record_duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file 


def play():
    sps = 44100
    sound_duration = 1.0
    freq_hz = 440.0
    atten = 0.3
    each_sample_number = np.arange(sound_duration * sps)
    waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
    waveform_quiet = waveform * atten
    sd.play(waveform_quiet, sps)
    time.sleep(sound_duration)
    sd.stop()

Thread(target = record).start()
time.sleep(1)
Thread(target = play).start()   