import numpy as np

words = set()
word_to_vec_map = {}

#opening text file which contains pre-trained word embeddings
with open('glove.6B.50d.txt', encoding='utf8') as f:
    
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)


def cosine_similarity(u, v):
    
    prod = np.dot(u, v)
    mag_u = np.sqrt(np.sum(np.square(u)))
    mag_v = np.sqrt(np.sum(np.square(v)))
    
    return prod/(mag_u*mag_v)

g = word_to_vec_map['woman'] - word_to_vec_map['man']


def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map[word]
    
    # Compute e_biascomponent using the formula give above. (≈ 1 line)
    e_biascomponent = np.dot(e, g)/np.dot(g, g)*g
 
    # Neutralize e by substracting e_biascomponent from it 
    # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
    e_debiased = e - e_biascomponent
    
    return e_debiased




def equalize(pair, bias_axis, word_to_vec_map):

    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    w1, w2 = pair[0], pair[1]
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2)/2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = np.dot(mu, bias_axis)/np.dot(bias_axis, bias_axis)*bias_axis
    mu_orth = mu - mu_B


    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
    e_w1B = np.dot(e_w1, bias_axis)/np.dot(bias_axis, bias_axis)*bias_axis
    e_w2B = np.dot(e_w2, bias_axis)/np.dot(bias_axis, bias_axis)*bias_axis
        
    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
    coeff = np.sqrt(np.abs((1-np.dot(mu_orth, mu_orth))))
    corrected_e_w1B = coeff * (e_w1B-mu_B)/np.abs(e_w1-mu_orth-mu_B)
    corrected_e_w2B = coeff * (e_w2B-mu_B)/np.abs(e_w2-mu_orth-mu_B)

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    
    return e1, e2



print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))






