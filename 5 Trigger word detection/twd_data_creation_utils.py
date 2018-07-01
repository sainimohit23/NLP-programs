import keras
import keras.backend as k
import matplotlib.pyplot as plt
import pydub
import numpy as np
import os
import scipy
from twd_data_read_utils import *



def get_random_time_segment(segment_ms):
    start = np.random.randint(0, 10000-segment_ms)
    end = start + segment_ms -1
    return (start, end)


def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    
    overlap = False
    for previous_start, previous_end in previous_segments:
        if previous_start<=segment_start<=previous_end or previous_start<=segment_end<=previous_end:
            overlap = True

    return overlap

def insert_audio_clips(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip) 
    time_seg = get_random_time_segment(segment_ms)
    
    while is_overlapping(time_seg, previous_segments) == True:
        time_seg = get_random_time_segment(segment_ms)
        
    previous_segments.append(time_seg)
    new_bg = background.overlay(audio_clip, position = time_seg[0])
    return new_bg, time_seg


def insert_ones(y, segment_end_ms, Ty):
    segment_end_y = int((segment_end_ms * Ty/10000.0))
    
    for i in range(segment_end_y, segment_end_y + 1):
        if i < Ty:
            y[0, i] = 1
            
    return y
    
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def create_training_examples(background, activates, negatives, Ty):
    
    background = background - 20
    y = np.zeros((1, Ty))
    previous_segments = []
    
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    for random_activate in random_activates:
        background, seg_time = insert_audio_clips(background, random_activate, previous_segments)
        segment_start, segment_end = seg_time
        y = insert_ones(y, segment_end, Ty)
    
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size = number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]
    
    for random_negative in random_negatives:
        background, _ = insert_audio_clips(background, random_negative, previous_segments)
        
        
    background = match_target_amplitude(background, -20)
    
    file_handle = background.export("train.wav", format="wav")
    x = graph_spectrogram("train.wav")
    
    return x, y
    
    
    
    
    
    
    
    
    
    








