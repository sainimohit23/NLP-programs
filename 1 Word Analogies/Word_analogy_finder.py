import numpy as np

words = set()
word_to_vec = {}

#opening text file which contains pre-trained word embeddings
with open('glove.6B.50d.txt', encoding='utf8') as f:
    
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)


def cosine_sim(u, v):
    
    prod = np.dot(u, v)
    mag_u = np.sqrt(np.sum(np.square(u)))
    mag_v = np.sqrt(np.sum(np.square(v)))
    
    return prod/(mag_u*mag_v)


def analogy(word_a, word_b, word_c, word_to_vec):
    
    e_a, e_b, e_c = word_to_vec[word_a.strip().lower()], word_to_vec[word_b.strip().lower()], word_to_vec[word_c.strip().lower()]
    words = word_to_vec.keys()
    max_similarity = -100
    best_fit_word = None
    
    for word in words:
        
        if word in [word_a, word_b, word_c]:
            continue
        
        sim = cosine_sim(e_b-e_a, word_to_vec[word] - e_c)
        
        if(sim > max_similarity):
            max_similarity = sim
            best_fit_word = word
            
    
    return best_fit_word


triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, analogy(*triad,word_to_vec)))