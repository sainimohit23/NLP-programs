import keras
import keras.backend as k
import matplotlib.pyplot as plt
import pydub
import numpy as np
import os
from scipy.io import wavfile


def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    
    for filename in os.listdir('raw_data/activates'):
        if filename.endswith(".wav"):
            activate = pydub.AudioSegment.from_wav("raw_data/activates/"+filename)
            activates.append(activate)
        
    for filename in os.listdir('raw_data/backgrounds'):
        if filename.endswith('.wav'):
            background = pydub.AudioSegment.from_wav("raw_data/backgrounds/"+filename)
            backgrounds.append(background)

    for filename in os.listdir("raw_data/negatives"):
        if filename.endswith('.wav'):
            negative = pydub.AudioSegment.from_wav("raw_data/negatives/"+filename)
            negatives.append(negative)
            
    return (activates, backgrounds, negatives)
            
            

"""COPIED"""
# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx



def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data