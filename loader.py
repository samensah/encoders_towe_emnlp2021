import json
import random
import torch
import numpy as np

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, args, file_path):
        self.batch_size = args.batch_size
        self.args = args
        self.file_path = file_path
        self.word2id = {w:i for i,w in enumerate(args.word_vocab)}

        with open(file_path, 'r') as f:
            self.raw_data = json.load(f)

        # give each instance a unique id to sample by id
        count = 0
        for d in self.raw_data:
            d['sample_id'] = count
            count += 1

        self.position2id, self.bio2id, self.rel2id, self.POS2id = args.position2id, args.bio2id, args.rel2id, args.POS2id
        self.data = self.preprocess(self.raw_data)
        self.num_examples = len(self.data)

        # chunk into batches
        self.data = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        print("{} batches created for {}".format(len(self.data), self.file_path))

    def padding_labels(self, labels, batch_size):
        """ Convert labels to a padded LongTensor. """
        token_len = max(len(x) for x in labels)
        padded_labels = torch.FloatTensor(batch_size, token_len, token_len).fill_(0)
        for i, s in enumerate(labels):
            padded_labels[i,:len(s),:len(s)] = torch.FloatTensor(s)
        return padded_labels

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []

        for d in data:
            # dict_keys(['tokens', 'POS', 'deprel', 'asp', 'opn'])
            tokens   = d['tokens']
            head     = d['head']
            # initialize an adjacency matrix for syntax dependencies among words
            deprel_m = [['[PAD]' for _ in range(len(tokens))] for _ in range(len(tokens))] 
            deprel   = d['deprel']
            for i in range(len(head)):
                if head[i] != 0:
                    deprel_m[head[i]-1][i] = deprel[i]
                    deprel_m[i][head[i]-1] = "r_"+deprel[i]
            POS      = d['POS']
            selfloop = ['selfloop' for _ in deprel]
            # elements of head start from 1
            
            # map to ids
            tokens   = map_to_ids(tokens, self.word2id)
            deprel   = [map_to_ids(row, self.rel2id) for row in deprel_m]
            selfloop = map_to_ids(selfloop, self.rel2id)
            POS      = map_to_ids(POS, self.POS2id)

            asp = d['asp'][0]
            asp_from = asp[0]
            asp_to   = asp[1]

            label = ['O' for _ in range(len(tokens))]
            for opn in d['opn']:
                if opn[0] == opn[1]:
                    label[opn[0]] = 'B'
                else:
                    label[opn[0]] = 'B'
                    for i in range(opn[0]+1, opn[1]+1, 1):
                        label[i] = 'I'
            label = map_to_ids(label, self.bio2id)
            
            position = [i-asp_from for i in range(asp_from)] + [0 for _ in range(asp_from, asp_to+1, 1)] + [i-asp_to for i in range(asp_to+1, len(tokens), 1)]
            position = map_to_ids(position, self.position2id)

            assert len(position) == len(tokens) == len(head) == len(deprel) == len(POS)

            l      = len(tokens)
            mask_s = [1 for i in range(l)]

            processed.append([d['sample_id'], tokens, position, POS, head, deprel, selfloop, mask_s, label])

        return processed

    def __len__(self):
        return len(self.data)

    # 0: tokens, 1: position, 2: POS, 3: head, 4: deprel, 5: selfloop, 6: mask_s, 7: label
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch)) # restructure batch input for pytorch
        # len batch -> len([d['sample_id'], tokens, position, POS, head, deprel, selfloop, mask_s, label])
        assert len(batch) == 9 

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[1]] 
        batch, _ = sort_all(batch, lens)

        # convert to tensors
        words       = get_long_tensor(batch[1], batch_size)
        position    = get_long_tensor(batch[2], batch_size)
        POS         = get_long_tensor(batch[3], batch_size)
        head        = get_long_tensor(batch[4], batch_size)
        deprel      = self.padding_labels(batch[5], batch_size).long()
        selfloop    = get_long_tensor(batch[6], batch_size)
        mask_s      = get_float_tensor(batch[7], batch_size)
        label       = get_float_tensor(batch[8], batch_size)
                
        return [batch[0], words, position, POS, head, deprel, selfloop, mask_s, label] 

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else vocab['[UNK]'] for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

