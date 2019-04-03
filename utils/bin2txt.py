"""
FROM THE GENSIM WORD2VEC DOCUMENTATION

It is impossible to continue training the vectors loaded from the C format 
because the hidden weights, vocabulary frequencies and the binary tree are 
missing. To continue training, you'll need the full 
:class:`~gensim.models.word2vec.Word2Vec` object state, as stored by 
:meth:`~gensim.models.word2vec.Word2Vec.save`, not just the 
:class:`~gensim.models.keyedvectors.KeyedVectors`.
"""

import gensim

# load Google's pre-trained Word2Vec model.
data = '../data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(data, binary=True)

# save vectors as text file
model.save_word2vec_format('../data/GoogleNews-vectors-negative300.txt', 
                           binary=False)