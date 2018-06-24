# This model outputs an emoji which best matches to what is described in a sequence.
# Model uses two layered LSTM network, see image for reference



import emoji_utils
import keras
import keras.backend as k
import numpy as np
import pandas as pd


train_data = pd.read_csv('train_emoji.csv')
X_train = train_data.iloc[:, 0].values
Y_train = train_data.iloc[:, 1].values

from sklearn.preprocessing import OneHotEncoder
Y_train = Y_train.reshape(Y_train.shape[0],1)
encoder = OneHotEncoder(categorical_features=[0])
Y_train = encoder.fit_transform(Y_train).toarray()


maxLen = len(max(X_train, key=len).split())
words_to_index, index_to_words, word_to_vec_map = read_glove_vectors('glove.6B.50d.txt')
m = X_train.shape[0]



# Creating embedding Layer for lstm 
def pretrained_embedding_layer(word_to_vec_map, words_to_index):
    emb_dim = word_to_vec_map['pen'].shape[0]
    vocab_size = len(words_to_index) + 1
    emb_matrix = np.zeros((vocab_size, emb_dim))
    
    for word, index in words_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    
    emb_layer= keras.layers.embeddings.Embedding(vocab_size, emb_dim, trainable= False)
    
    emb_layer.build((None,))
    emb_layer.set_weights([emb_matrix])
    
    return emb_layer


# LSTM emojify model
def Emojify(input_shape, word_to_vec_map, words_to_index):
    
    sentance_indices = keras.layers.Input(shape = input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)
    embeddings = embedding_layer(sentance_indices)
    
    X = keras.layers.LSTM(128, return_sequences=True)(embeddings)
    X = keras.layers.Dropout(0.5)(X)
    
    X = keras.layers.LSTM(128, return_sequences=False)(X)
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.Dense(5)(X)
    
    X = keras.layers.Activation('softmax')(X)
    
    model = keras.models.Model(sentance_indices, X)
    
    return model
    
        
model = Emojify((maxLen,), word_to_vec_map,words_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentance_to_indices(X_train, words_to_index, maxLen)
model.fit(X_train_indices, Y_train, epochs = 50, batch_size = 32, shuffle=True)








"""Predictions on test data"""
test_data = pd.read_csv('tesss.csv')
X_test = test_data.iloc[:, 0].values
Y_test = test_data.iloc[:, 1].values

from sklearn.preprocessing import OneHotEncoder
Y_test = Y_test.reshape(Y_test.shape[0],1)
encoder = OneHotEncoder(categorical_features=[0])
Y_test = encoder.fit_transform(Y_test).toarray()

X_test_indices = sentance_to_indices(X_test, words_to_index, maxLen)

predictions = model.predict(X_test_indices)

for i in range(len(predictions)):
    pred = np.argmax(predictions[i])
    seq = X_test[i]
    
    print(seq + label_to_emoji(pred))    




















