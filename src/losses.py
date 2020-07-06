# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.utils import one_hot_label
from config import Config

CONFIG = Config
DEVICE = CONFIG.device

# FOCAL LOSS CLASS
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reductio=True):
      super(FocalLoss, self).__init__()
      self.alpha = alpha
      self.gamma = gamma
      self.logits = logits
      self.reduction = reductio

    def forward(self, inputs, targets):
      BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
      pt = torch.exp(-BCE_loss)
      F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

      if self.reduction:
          return torch.mean(F_loss)
      else:
          return F_loss

# CLASS BALANCED LOSS
class CB_Loss(nn.Module):
    def __init__(self, sample_per_cls, no_of_cls, beta, loss='ce', gamma=2):
      super(CB_Loss, self).__init__()
      self.sample_per_cls = sample_per_cls
      self.gamma = gamma
      self.no_of_cls = no_of_cls
      self.beta = beta
      self.gamma = gamma
      self.loss = loss

    def forward(self, inputs, target):
      effective_num = 1.0 - np.power(self.beta, self.sample_per_cls)
      weights = (1.0 - self.beta) / np.array(effective_num)
      weights = weights / np.sum(weights) * self.no_of_cls
      weights = torch.tensor(weights).to(DEVICE)

      if self.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(inputs, target)
        return loss
      if self.loss == 'fl':
        criterion = FocalLoss(alpha=weights)
        loss = criterion(inputs, target)
        return loss

    