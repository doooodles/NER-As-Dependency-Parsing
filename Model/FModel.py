from abc import ABC

from transformers import XLNetModel, BertModel

from Layers import feedforwardLayer, biaffineLayer
import torch.nn as nn
import torch.nn.functional as F
import torch


class FModel(nn.Module, ABC):

    def __init__(self, d_in, d_hid, d_class, n_layers, bi=True, dropout=0.3):
        super().__init__()
  