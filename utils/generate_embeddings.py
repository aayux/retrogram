"""
Convert pretrained GloVe embeddings into npy file
"""

import numpy as np
import pickle
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--npy_output', type=str, required=True)
    parser.add_argument('--dict_output', type=str, required=True)
    parser.add_argument('--dump_frequency', type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()

    data = {}
    embeddings = []

    float_re = re.compile(r' [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')

    print('Building vocabulary. This may take a while ...')

    with open(args.dataset) as ofile, \
         open(args.dict_output, 'wb') as dfile, \
         open(args.npy_output, 'wb') as nfile:
        idx = 1
        for line in ofile:
            pos = next(re.finditer(float_re, line)).start()
            word, vector = line[:pos], line[pos + 1:].split()

            if word in data:
                print(f'Possible duplicate at {idx} in {line}')
                continue
            
            embedding = np.fromiter([float(d) for d in vector], np.float32)
            
            if embedding.shape != (300,):
                print(f'Shape is {embedding.shape}')
                print(line)
            
            embeddings.append(embedding)
            data[word] = idx

            idx += 1
            
            if not idx % args.dump_frequency:
                np.save(nfile, np.array(embeddings))
                embeddings.clear()

        np.save(nfile, np.array(embeddings))
        pickle.dump(data, dfile)

    print(f'Vocabulary saved, size is {idx} words')

if __name__ == '__main__':
    main()