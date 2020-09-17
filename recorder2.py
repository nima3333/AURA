#https://realpython.com/playing-and-recording-sound-python/#recording-audio
#https://gist.github.com/akey7/94ff0b4a4caf70b98f0135c1cd79aff3

#Do not work

import numpy as np
import sounddevice as sd
import sounddevice as sd2
import time
from scipy.io.wavfile import write
from threading import Thread
import pygame
import soundfile as sf

def record():
    fs = 44100
    record_duration = 5
    myrecording = sd.rec(int(record_duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    #write('output.wav', fs, myrecording)  # Save as WAV file 
    sf.write('myfile.wav', int(record_duration * fs), fs, subtype='PCM_24')


def play():
    sps = 44100
    sound_duration = 1.5
    freq_hz = 440.0
    atten = 0.6
    each_sample_number = np.arange(sound_duration * sps)
    waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
    waveform_quiet = waveform * atten
    x  = (waveform_quiet*32768).astype(np.int16)
    pygame.mixer.pre_init(sps, size=-16, channels=1)
    pygame.mixer.init()
    sound = pygame.sndarray.make_sound(x)
    sound.play()
    time.sleep(sound_duration)

time.sleep(8)
#a = Thread(target = record)
b = Thread(target = play)

#a.start()
b.start()

b.join()
#a.join()