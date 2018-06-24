import emoji
import numpy as np
import pandas as pd

emoji_dictionary = {"0": "\u2764\uFE0F",    
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}


# Do as function name says
def label_to_emoji(index):   
    return emoji.emojize(emoji_dictionary[str(index)], use_aliases=True)


# Reads embeddings and return 3 dictionaries
def read_glove_vectors(path):
    with open(path, encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            cur_word = line[0]
            words.add(cur_word)
            word_to_vec_map[cur_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
    



# Convert sentance words to thier corresponding indices and adds zero padding
def sentance_to_indices(X, words_to_index, maxLen):
    m = X.shape[0]    
    X_indices = np.zeros((m, maxLen))
    
    for i in range(m):
        sentance_words = X[i].lower().strip().split()
        
        j = 0
        for word in sentance_words:
            X_indices[i, j] = words_to_index[word]
            j += 1
            
    return X_indices








