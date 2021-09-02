"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
import random
from collections import Counter

random.seed(1234)
np.random.seed(1234)

VOCAB_PREFIX = ['[PAD]', '[UNK]']

def parse_args():
    parser = argparse.ArgumentParser(description="Gnerate vocabulary for the atis or snips.")
    parser.add_argument('--dataset', default='16res', help='dataset directory atis or snips.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    train_file  = 'data/'+args.dataset+'/train.json'
    test_file   = 'data/'+args.dataset+'/test.json'

    wv_file = 'data/glove/'+args.wv_file
    wv_dim  = args.wv_dim

    vocab_file = 'data/'+args.dataset+'/vocab.pkl'
    emb_file   = 'data/'+args.dataset+'/embedding.npy'

    print("loading tokens...")
    train_tokens  = load_tokens(train_file)
    test_tokens   = load_tokens(test_file)
    
    print("loading glove words...")
    glove_vocab = load_glove_vocab(args.wv_file, args.wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens+test_tokens, glove_vocab)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))

    print("building embeddings...")
    embedding = build_embedding(args.wv_file, v, args.wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)

    print("all done.")

def load_glove_vocab(filename, wv_dim):
    vocab = set()
    with open('./data/glove/'+filename, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab

def load_tokens(file_path):
    data = read_json(file_path)
    tokens = []
    for d in data:
        ts = d['tokens']
        tokens += list(ts)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), file_path))
    return tokens

def build_vocab(tokens, glove_vocab):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # sort words according to its freq
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens
    v = VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # pad vector
    w2id = {w: i for i, w in enumerate(vocab)}
    with open('./data/glove/'+wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def read_json(file_path):
    with open(file_path, 'r') as f:
            data = json.load(f)
    return data

if __name__ == '__main__':
    main()


