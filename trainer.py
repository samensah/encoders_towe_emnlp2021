import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model import *

class Trainer(object):
    def __init__(self, args, embedding_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

# 0: tokens, 1: mask_sent, 2: ote_labels, 3: opn_labels, 4: ts_labels
def unpack_batch(batch):
    for i in range(len(batch)):
        batch[i] = Variable(batch[i].cuda())
    return batch

# 0: tokens, 1: mask_sent, 2: ote_labels, 3: opn_labels, 4: ts_labels
class MyTrainer(Trainer):
    def __init__(self, args):
        self.args             = args
        self.model            = Toy_model(args).cuda()
        self.parameters       = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer        = torch.optim.Adam(self.parameters, lr=args.lr)

        print(np.sum([p.numel() for p in self.parameters]).item())

    def update(self, batch):
        batch = unpack_batch(batch[1:])
        # step forward
        self.model.train()
        loss, _ = self.model(batch)


        # backward of task loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss value
        loss = loss.item()

        return loss 

    def predict(self, batch):
        with torch.no_grad():
            batch = unpack_batch(batch[1:])
            # forward
            self.model.eval()
            loss, pred = self.model(batch)

        return loss, pred
