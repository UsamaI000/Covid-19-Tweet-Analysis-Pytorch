# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from src.utils import one_hot_label

is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda")
else: device = torch.device("cpu")

# FOCAL LOSS CLASS
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction= 'sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        targets = one_hot_label(targets, 5)
        y = targets.to(device)
        alpha = self.alpha
        gamma = self.gamma
        if self.reduction == 'mean':
            return -torch.mean(y*alpha*(1-p)**gamma*torch.log(p) + (1-y)*(1-alpha)*(p)**gamma*torch.log(1-p))
        elif self.reduction == 'sum':
            return -torch.sum(y*alpha*(1-p)**gamma*torch.log(p) + (1-y)*(1-alpha)*(p)**gamma*torch.log(1-p))
        else:
            return  -1* (y*alpha*(1-p)**gamma*torch.log(p) + (1-y)*(1-alpha)*(p)**gamma*torch.log(1-p))