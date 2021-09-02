"""
models for sentiment.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tree import *


# 0: tokens, 1: position, 2: POS, 3: head, 4: deprel, 5: selfloop, 6: mask_s,  7: label
class Toy_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_matrix = args.embedding_matrix
        
        self.encoder = LSTMRelationModel(args)
        self.gcns    = nn.ModuleList([nn.Linear(2*args.dim_bilstm_hidden, 2*args.dim_bilstm_hidden) for _ in range(args.gcn_layers)])

        # create embedding layers
        self.emb = nn.Embedding(args.vocab_size, args.dim_w, padding_idx=0)
        self.init_embeddings()
        self.emb.weight.requires_grad = True 
        if args.dim_POS != 0:
            self.POS_emb = nn.Embedding(len(args.POS2id), args.dim_POS, padding_idx=0)
        if args.dim_position != 0:
            self.position_emb = nn.Embedding(len(args.position2id), args.dim_position, padding_idx=0)
        if args.dim_deprel != 0:
            self.deprel_emb = nn.Embedding(len(args.rel2id), args.dim_deprel, padding_idx=0) 

        # dropout
        self.input_dropout = nn.Dropout(args.dropout_rate)

        # classifer
        self.classifier   = nn.Linear(2*args.dim_bilstm_hidden, len(args.bio2id))

        # loss function 
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none')

    def init_embeddings(self):
        if self.embedding_matrix is not None:
            self.embedding_matrix = torch.from_numpy(self.embedding_matrix)
            self.emb.weight.data.copy_(self.embedding_matrix)

    # [words, position, POS, head, deprel, selfloop, mask_s, label]
    def forward(self, inputs):
        
        # Bilstm encoder
        tokens, position, POS, head, deprel, selfloop, mask_s, label  = inputs
        
        tokens_embs = self.emb(tokens)
        if self.args.dim_POS != 0:
            tokens_embs = torch.cat([tokens_embs, self.POS_emb(POS)], dim=2)
        if self.args.dim_position != 0:
            tokens_embs = torch.cat([tokens_embs, self.position_emb(position)], dim=2)
        deprel_embs   = self.deprel_emb(deprel)
        selfloop_embs = self.deprel_emb(selfloop)

        lens = mask_s.sum(dim=1).cpu()
        rnn_inputs = self.input_dropout(tokens_embs)
        H          = self.encoder((rnn_inputs, lens)) 

        # adj
        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            maxlen = int(max(l))
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda()

        adj = inputs_to_tree_reps(head.tolist(), lens.tolist())
        
        # GCN layers
        denom = adj.sum(2).unsqueeze(2) + 1    # norm
        for gcn in self.gcns:
            Ax   = adj.bmm(H)
            AxW  = gcn(Ax)
            AxW  = AxW / denom
            gAxW = F.relu(AxW)
            H    = gAxW + H
        
        logits = self.classifier(H)
        # pred and loss
        opn_pred = torch.argmax(logits, dim=2)
        opn_pred = opn_pred * mask_s.long()
        opn_loss = self.ce_loss(logits.reshape(-1, logits.size(-1)), label.long().reshape(-1))
        opn_loss = opn_loss.reshape(logits.size(0), logits.size(1))
        opn_loss = (opn_loss * mask_s).sum() / opn_loss.size(0)

        return opn_loss, opn_pred.tolist()

# BiLSTM model 
class LSTMRelationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.dim_w + args.dim_POS + args.dim_position
        self.rnn = nn.LSTM(self.in_dim, args.dim_bilstm_hidden, 1, batch_first=True, dropout=0.0, bidirectional=True)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.dim_bilstm_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        # unpack inputs
        inputs, lens = inputs[0], inputs[1]
        return self.encode_with_rnn(inputs, lens, inputs.size()[0])

# Initialize zero state
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class GCNRelationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim_bilstm_hidden*2
        self.layers = args.gcn_layers
        self.gcn_drop = nn.Dropout(args.gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()
        for i in range(self.layers):
            self.W.append(nn.Linear(self.dim, self.dim))

    def forward(self, inputs, nondia_adj, dia_adj):
        
        # mask inputs   #50x65x1
        inputs = inputs*dia_adj.sum(dim=2).unsqueeze(-1)#H0 = Ad * X   [50, 65, 300]

        # forward GCN layers
        denom = nondia_adj.sum(2).unsqueeze(2) + 1
        gcn_inputs = inputs
        for l in range(self.layers):
            tmp_inputs = gcn_inputs
            Ax = nondia_adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom 
            gAxW = F.relu(AxW)
            gcn_inputs = gAxW + tmp_inputs

            if l == 0 and l < self.layers - 1:
                gcn_inputs = self.gcn_drop(gcn_inputs) 
                        
        outputs = gcn_inputs #[32, 65, 600]
        return outputs


