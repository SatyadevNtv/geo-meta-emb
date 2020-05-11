import os
import gc
import argparse
import numpy as np

import embeddings
from geomm import normalize_emb

class Embedding(object):
    """
    Store the words, emb data
    """

    def __init__(self, words, vecs):
        if len(list(set(words))) != len(words):
            raise RuntimeError("Only unique words list supported")
        self.word_vec_map = {x[0]: x[1] for x in zip(words, vecs)}

    def get(self, word):
        return self.word_vec_map[word]

def intersection(e1, e2):
    """
    Find the common words between 2 embeddings
    """
    return list(set(list(e1.word_vec_map.keys())) & set(list(e2.word_vec_map.keys())))


def avg(e1, e2):
    """
    Average the input embedding's vectors per word
    """
    words = intersection(e1, e2)
    print(f"Common word count: {len(words)}")
    vecs = np.vstack((e1.get(w) + e2.get(w))/2 for w in words)
    return Embedding(words, vecs)


def concatenate(e1, e2):
    """
    Concatenate the input embedding's vectors per word
    """
    words = intersection(e1, e2)
    print(f"Common word count: {len(words)}")
    vecs = np.vstack(np.concatenate((e1.get(w), e2.get(w))) for w in words)
    return Embedding(words, vecs)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate meta embeddings')
    parser.add_argument('emb1', help='path to embedding 1')
    parser.add_argument('emb2', help='path to embedding 2')
    parser.add_argument('--method', choices=['avg', 'conc'],default=['avg'], type=str, nargs=1, help='meta embedding generation method')
    parser.add_argument('--meta_embeddings_path', default='./', type=str, help='directory to save the output meta embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--verbose', default=0,type=int, help='Verbose')
    parser.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'no'], nargs=2, default=[], help='the normalization actions performed in sequence for embeddings 1 and 2')


    args = parser.parse_args()
    
    if args.verbose:
        print('Current arguments: {0}'.format(args))

    dtype = 'float32'
    if args.verbose:
        print('Loading embeddings data...')

    emb1file = open(args.emb1, encoding=args.encoding, errors='surrogateescape')
    emb2file = open(args.emb2, encoding=args.encoding, errors='surrogateescape')
    emb1_words, x = embeddings.read(emb1file,max_voc=0, dtype=dtype)
    emb2_words, z = embeddings.read(emb2file,max_voc=0, dtype=dtype)

    if len(args.normalize) > 0:
        x = normalize_emb(x, args.normalize[0])
        z = normalize_emb(z, args.normalize[1])

    emb1 = Embedding(emb1_words, x)
    emb2 = Embedding(emb2_words, z)

    if args.method[0] == "avg":
        meta_emb = avg(emb1, emb2)
    elif args.method[0] == "conc":
        meta_emb = concatenate(emb1, emb2)

    del emb1, emb2
    gc.collect()

    meta_emb_words = []
    meta_emb_vecs = []
    for w, v in meta_emb.word_vec_map.items():
        meta_emb_words += [w]
        meta_emb_vecs += [v]

    del meta_emb
    gc.collect()

    out_emb_fname=os.path.join(args.meta_embeddings_path,'meta_emb.vec')
    with open(out_emb_fname,'w',encoding=args.encoding) as outfile:
        embeddings.write(meta_emb_words,meta_emb_vecs,outfile)


if __name__ == "__main__":
    main()
