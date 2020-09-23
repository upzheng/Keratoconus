import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class BaseNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @abstractmethod
    def model_name(self):
        raise NotImplementedError

    def save_model(self, step):

        save_fname = '%s_%s.pth' % (step, self.model_name())
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        save_dict = {
            'net_state_dict': self.state_dict(),
            'exp_name': self.args.exp_name,
            'iter': step
        }
        torch.save(save_dict, save_path)
        print('Model Saved')

    def save_model_best(self, step, best='best_recall', index=None):
        save_fname = '%s_%s_%s.pth' % (self.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        save_dict = {
            'net_state_dict': self.state_dict(),
            'exp_name': self.args.exp_name,
            'iter': step,
            'index': index
        }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')

    def load_model_best(self, model_path):
        if os.path.exists(model_path):
            load_dict = torch.load(model_path)
            net_state_dict = load_dict['net_state_dict']

            self.load_state_dict(net_state_dict)
            self.iter = load_dict['iter'] + 1
            index = load_dict['index']

            print('Model Loaded!')
            return self.iter, index
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def load_model(self, model_path):

        assert model_path.endswith('.pth'), 'Only .pth files supported.'

        print('Loading weights into state dict from {}...'.format(model_path))
        load_dict = torch.load(model_path)
        net_state_dict = load_dict['net_state_dict']

        self.load_state_dict(net_state_dict)
        self.iter = load_dict['iter'] +1

        print('Model Loaded!')
        return self.iter

    def predict(self, x, thresh=0.5):
        self.eval()
        score = self(x)
        score = F.softmax(score, dim=1)
        ind, pred = score.max(1)
        #pred = (score[:, 1] > thresh).int()
        return pred, ind, score  # score is increased by up
