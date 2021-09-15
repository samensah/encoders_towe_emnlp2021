import os
import shutil
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import pickle
import copy
import json

from config import *
from evals import *
from loader import DataLoader 
from trainer import MyTrainer
import time 

seed = random.randint(1, 10000)

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)


# load vocab and embedding matrix
dataset_path          = "data/%s"        % (args.dataset)
vocab_path            = "%s/vocab.pkl"        % dataset_path
embedding_path        = "%s/embedding.npy"    % dataset_path
print('loading vocab and embedding matrix from {}'.format(dataset_path))
with open(vocab_path, 'rb') as f:
    word_vocab = pickle.load(f)
args.word_vocab = word_vocab
embedding_matrix = np.load(embedding_path)
args.embedding_matrix = embedding_matrix
assert embedding_matrix.shape[0] == len(word_vocab)
assert embedding_matrix.shape[1] == args.dim_w
print('size of vocab: {}'.format(len(word_vocab)))
print('shape of loaded embedding matrix: {}'.format(embedding_matrix.shape))
args.vocab_size = len(word_vocab)

# load data
train_path  = '%s/train.json' % (dataset_path)
test_path   = '%s/test.json'  % (dataset_path)

print("Generating mappings to transform inputs into sequence of ids")
# generate POS2id, bio2id, position2id, rel2id
args.bio2id = {'O':0, 'B':1, 'I':2}
POS_list, rel_list = ['[PAD]'], ['[PAD]', 'selfloop']
max_len = -1

with open(train_path, 'r') as f:
    raw_train = json.load(f)
with open(test_path, 'r') as f:
    raw_test = json.load(f)
raw_data = raw_train + raw_test

for d in raw_data:
    for POS in d['POS']:
        if POS not in POS_list:
            POS_list.append(POS)

    for rel in d['deprel']:
        if rel not in rel_list:
            rel_list.append(rel)
            rel_list.append("r_"+rel) # adding a reverse relation

    if len(d['tokens']) > max_len:
        max_len = len(d['tokens'])

position_list = []
for i in range(1, max_len+1, 1):
    position_list.append(i)
    position_list.append(-i)
position_list = ['[PAD]', 0] + position_list
args.position2id = {p:i for i,p in enumerate(position_list)}

args.POS2id = {p:i for i,p in enumerate(POS_list)}
args.rel2id = {p:i for i,p in enumerate(rel_list)}


print("Loading data from {} with batch size {}...".format(dataset_path, args.batch_size))
train_batches  = DataLoader(args, train_path)
test_batches   = DataLoader(args, test_path)


# create the folder for saving the best model
if os.path.exists(args.save_dir) != True:
    os.mkdir(args.save_dir)

log_file = FileLogger(args.save_dir+"/log.txt")

print('Building model...')
# create model
trainer_model  = MyTrainer(args)

# start training
estop      = 0
batch_num  = len(train_batches)
current_best_F1 = -1
for epoch in range(1, args.n_epoch+1):
    starttime = time.time()
    
    if estop > args.early_stop:
        break

    train_loss, train_step = 0., 0
    for i in range(batch_num):
        batch = train_batches[i]
        loss = trainer_model.update(batch)
        train_loss += loss
        train_step += 1
        
        # print training loss 
        if train_step % args.print_step == 0:
            print("[{}] train_loss: {:.4f}".format(epoch, train_loss/train_step))
    endtime = time.time()
    print((endtime - starttime))
    
    # evaluate on unlabel set
    print("")
    print("Evaluating...Epoch: {}".format(epoch))
    eval_scores, eval_loss = evaluate_program(trainer_model, test_batches, args)
    print("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))
    # loging
    log_file.log("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))

    if eval_scores[-1] > current_best_F1:
        current_best_F1 = eval_scores[-1]
        trainer_model.save(args.save_dir+'/best_model.pt')
        print("New best model saved!")
        log_file.log("New best model saved!")
        estop = 0

    estop += 1
    print("")


print("Training ended with {} epochs.".format(epoch))

# final results
trainer_model.load(args.save_dir+'/best_model.pt')
eval_scores, eval_loss = evaluate_program(trainer_model, test_batches, args)

print("Final result:")
print("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))

# loging
log_file.log("Final result:")
log_file.log("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))
