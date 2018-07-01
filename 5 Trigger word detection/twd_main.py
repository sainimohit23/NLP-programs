import keras
import keras.backend as k
import matplotlib.pyplot as plt
import pydub
import numpy as np
import os
import scipy
from twd_data_creation_utils import *
from twd_data_read_utils import *
import IPython

activates, backgrounds, negatives = load_raw_audio()

#Specgram shapes
shape_specctrogram = graph_spectrogram("audio_examples/example_train.wav")
Tx = shape_specctrogram.shape[1]
n_f = shape_specctrogram.shape[0]

#original shapes
rate, data = scipy.io.wavfile.read("audio_examples/example_train.wav")

#output shape
Ty = 1375

x, y = create_training_examples(backgrounds[0], activates, negatives, Ty)




"""""""""                           MODEL                          """""""""


# Loading pretrained train and dev data
X = np.load("XY_train/X.npy")
Y = np.load("XY_train/Y.npy")

X_dev = np.load("XY_dev/X_dev.npy")
Y_dev = np.load("XY_dev/Y_dev.npy")


def create_model(input_shape):
    
    X_input = keras.layers.Input(shape=input_shape)
    X = keras.layers.Conv1D(196, 15, strides=4)(X_input)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.8)(X)
    
    X = keras.layers.GRU(128, return_sequences=True)(X)
    X = keras.layers.Dropout(0.8)(X)
    X = keras.layers.BatchNormalization()(X)
    
    X = keras.layers.GRU(units = 128, return_sequences = True)(X)   
    X = keras.layers.Dropout(0.8)(X)                                 
    X = keras.layers.BatchNormalization()(X)                                 
    X = keras.layers.Dropout(0.8)(X)
    
    X = keras.layers.TimeDistributed(keras.layers.Dense(1,activation='sigmoid'))(X)
    
    model = keras.models.Model(X_input, X)
    
    return model


model = create_model((Tx, n_f))




"""""""""                   Loaing Pretrained Model                 """""""""

model = keras.models.load_model('./models/tr_model.h5')
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


def make_predictions(filename):
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions




chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = pydub.AudioSegment.from_wav(filename)
    chime = pydub.AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
 
    consecutive_timesteps = 0
   
    for i in range(Ty):
        
        consecutive_timesteps += 1
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')

filename = "./raw_data/dev/2.wav"
prediction = make_predictions(filename)
chime_on_activate(filename, prediction, 0.5)






