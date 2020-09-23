import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss


class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, score, target):

        target = target.data.cpu()
        target_onehot = torch.zeros(score.size())
        target_onehot.scatter_(1, target.view(target.size(0), -1), 1.0)
        target_onehot = Variable(target_onehot.cuda())

        score = F.softmax(score)
        pt = target_onehot * score + (1 - target_onehot) * (1 - score)
        return torch.sum(-(1-pt).pow(2) * torch.log(pt), 1).mean()


class nllloss(nn.Module):

    def __init__(self, weight=None):
        super(nllloss, self).__init__()
        self.mlay = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss(weight=weight)

    def forward(self, score, target):
        output = self.loss(self.mlay(score), target)

        return output


class nll_CEloss(nn.Module):

    def __init__(self, weight=None):
        super(nll_CEloss, self).__init__()
        self.mlay = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss(weight=weight)
        self.loss2 = CrossEntropyLoss(weight=weight)

    def forward(self, score, target):
        output = self.loss(self.mlay(score), target) + self.loss2(score, target)

        return output

