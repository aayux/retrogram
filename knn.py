import os
import fire

import numpy as np
import pickle as pkl

from sklearn.neighbors import NearestNeighbors
from utils.data_utils import idx_to_word

def most_similar(knn, word_embedding, vocab):
    nn_idxs = knn.kneighbors([word_embedding], return_distance=False).squeeze()
    nn_words = [idx_to_word(idx, vocab) for idx in nn_idxs]
    return nn_words


def nearest_neighbor_search(word, k, directory, experiment='base', path='./data', metric='cosine'):
    embeddings = np.load(os.path.join(path, directory, 'embeddings', f'{experiment}.npy'))
    vocab = pkl.load(open(os.path.join(path, directory, 'vocab', f'{experiment}.pkl'), 'rb'))

    try: 
        word_idx = vocab[word]

    except KeyError:
        print('Word not in vocabulary.')
    
    knn = NearestNeighbors(k, metric=metric)
    knn.fit(embeddings)
    
    word_embedding = embeddings[word_idx]
    print(most_similar(knn, word_embedding, vocab))

if __name__ == '__main__': 
    fire.Fire(nearest_neighbor_search)