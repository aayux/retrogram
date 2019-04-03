import os
import gc
import random
import itertools
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import Counter

def word_to_idx(words, vocab): 
    r""" Replace all the words in dataset with the idx in the vocab
    """
    return [vocab[word] if word in vocab else vocab['unk'] for word in words]

def idx_to_word(idx, vocab):
    return list(vocab.keys())[idx]

def iterate(word_idxs, context_window):
    r""" Generate training pairs according to skip-gram model
    """
    # TO DO: Subsampling
    #
    for idx, word in enumerate(word_idxs):
        context = random.randint(1, context_window)
        for targ in word_idxs[max(0, idx - context) : idx]:
            yield word, targ
        for targ in word_idxs[idx + 1 : idx + context + 1]:
            yield word, targ

def batchify(iterator, batch_size):
    r""" Generate batches and return them as numpy arrays
    """
    while True:
        word_batch = np.zeros((batch_size), dtype=np.int32)
        targ_batch = np.zeros((batch_size, 1))
        for idx in range(batch_size):
            word_batch[idx], targ_batch[idx] = next(iterator)
        yield word_batch, targ_batch

def p(f, t=.5):
    r""" subsampling formula
    """
    if f:
        return (t / f) ** .5
    return 1.

class DataLoader(object):
    def __init__(self, path, directory, experiment):
        self.path = path
        self.experiment = experiment
        self.directory = directory
        self.base_vocab = None
    
    def word2phrase(self, words, delta=6, threshold=5e-3):        
        sentences = ' '.join(words).split(' </s> ')
        phrases = []

        for sentence in sentences:
            word = sentence.split()
            for _w, w_ in zip(word, word[1:]):
                phrase = f'{_w}_{w_}'
                phrases.append(phrase)
        
        word_count = []
        phrase_count = []

        word_count.extend(Counter(words).most_common())
        phrase_count.extend(Counter(phrases).most_common())

        word_count = dict(word_count)
        phrase_count = dict(phrase_count)

        for idx, sentence in enumerate(tqdm(sentences)):
            word = sentence.split()
            for ix, (_w, w_) in enumerate(zip(word, word[1:])):
                if _w != '</r>' and _w != '.' and w_ != '</r>' and w_ != '.':
                    phrase = f'{_w}_{w_}'
                    score = (phrase_count[phrase] - delta) / (word_count[_w] * word_count[w_])
                    if score >= threshold:
                        word[ix] = phrase
                        word[ix + 1] = '</r>'
            sentences[idx] = ' '.join(word)
        words = ' '.join(sentences).split()
        return words

    def reader(self, file):
        r""" Reads data from the text file.
        """
        with open(file) as f:
            words = list(map(lambda x: x.lower(), f.read().split()))
            # tf.compat.as_str() converts the input into the string
        return words
    
    def build_data(self, words, top_k, dims):
        r""" Builds vocabulary of vocab_size from the words and generates tthe corresponding word
             embeddings.
        """
        
        old_vocab = []
        new_vocab = []
        
        # cast all vocab keys to lower case
        # reversed because we want the first occurence of the word
        base_vocab = {key.lower(): value for key, value in reversed(list(self.base_vocab.items()))}

        word_list = [('unk', 0)]
        embeddings = []

        word_list.extend(Counter(words).most_common(top_k))

        for word, count in word_list:
            # create new vocab and subsample frequent words
            # if random.random() < p(count):
            if word in base_vocab:
                old_vocab.append(word)
                base_idx = base_vocab[word] - 1
                embeddings.extend([self.base_embeddings[base_idx]])
            else:
                new_vocab.append(word)

        del self.base_embeddings
        gc.collect()
        
        embeddings = np.row_stack((np.random.uniform(low=-1., high=1., 
                                                        size=(len(new_vocab), dims)), 
                                      embeddings))

        vocab = new_vocab + old_vocab
        vocab = dict(zip(vocab, list(range(0, len(vocab)))))
        
        if not os.path.exists(os.path.join(self.path, self.directory, 'vocab')):
            os.makedirs(os.path.join(self.path, self.directory, 'vocab'))

        pkl.dump(vocab, open(os.path.join(self.path, self.directory, 'vocab', 
                                          f'{self.experiment}.pkl'), 'wb'))

        return vocab, embeddings, len(new_vocab)

    def prepare(self, top_k, dims=300):
        words = self.reader(os.path.join(self.path, self.directory, 'text'))

        # bi-gram phrase detection
        words = self.word2phrase(words)
        
        # tri-gram phrase detection
        words = self.word2phrase(words, delta=8)

        words = list(filter(lambda x: x != '</r>', words))

        vocab, embeddings, slice_ = self.build_data(words, top_k, dims)
        word_idxs = word_to_idx(words, vocab)
        return word_idxs, embeddings, slice_

    def load_word2vec(self, filename, dims=300):
        self.base_vocab = pkl.load(open(os.path.join(self.path, 'base', 'vocab.pkl'), 'rb'))
        vocab_size = len(self.base_vocab)

        embedding_matrix = np.zeros((vocab_size, dims), dtype=np.float32)
        
        # As embedding matrix could be quite big we 'stream' it into output file
        # chunk by chunk. One chunk shape could be [size // 10, dims].
        # So to load whole matrix we read the file until it's exhausted.
        path = os.path.join(self.path, 'base', filename)
        size = os.stat(path).st_size
        with open(path, 'rb') as ifile:
            pos = 0
            idx = 0
            while pos < size:
                chunk = np.load(ifile)
                chunk_size = chunk.shape[0]
                embedding_matrix[idx : idx + chunk_size, :] = chunk
                idx += chunk_size
                pos = ifile.tell()
        
        self.base_embeddings = embedding_matrix
        return