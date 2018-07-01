import keras
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


m = 10000 # number of training examples
dataset, human_vocab, machine_vocab, inverse_machine_vocab = load_dataset(m) # creates artificial dataset


Tx = 30 #input timestamps
Ty = 10 #output timestamps
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)


repeater = keras.layers.RepeatVector(Tx)
concatenator = keras.layers.Concatenate(axis=-1)
densor1 = keras.layers.Dense(10, activation='tanh')
densor2 = keras.layers.Dense(1, activation='relu')
activator = keras.layers.Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = keras.layers.Dot(axes = 1)

def one_step_attention(a, s_prev):
    s_prev = repeater(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    
    return context

n_a = 32
n_s = 64
post_activation_LSTM_cell = keras.layers.LSTM(n_s, return_state = True)
output_layer = keras.layers.Dense(len(machine_vocab), activation=softmax)


def model(n_s, n_a, Tx, Ty, human_vocab_size, machine_vocab_size):
    
    X =  keras.layers.Input(shape=(Tx, human_vocab_size))
    s0 =  keras.layers.Input(shape=(n_s,), name='s0')
    c0 =  keras.layers.Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    
    a = keras.layers.Bidirectional(keras.layers.LSTM(n_a, return_sequences=True))(X)
    
    for i in range(Ty):
        context = one_step_attention(a, s)      
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        
        out = output_layer(s)
        outputs.append(out)
    
    model = keras.models.Model([X, s0, c0], outputs)
    return model


model = model(n_s, n_a, Tx, Ty, len(human_vocab), len(machine_vocab))
opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)








