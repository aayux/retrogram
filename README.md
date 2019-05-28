# Retrogram

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Retrofitted word vectors with Word2Vec skip-gram model

This project is inspired from [Mittens](https://github.com/roamanalytics/mittens) which extends the GloVe model 
to synthesize general-purpose representations with specialised datasets. The resulting representations of words 
are arguably more context aware than the pre-trained embeddings. 

However, GloVe objective requires a co-occurence matrix of size V<sup>2</sup> to be 
held in-memory, where V is the size of the domain adapted vocabulary. Needless to say, this method 
becomes difficult to scale with growing vocabulary size.

Replacing the GloVe model with skip-gram reduces the size of the matrix to V×E where E is the embedding dimension 
and depends on the pre-trained word-embeddings being utilised.
