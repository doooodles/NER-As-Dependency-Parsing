from abc import ABC

import torch.nn as nn
import torch.nn.functional as F
import torch


class feedforwardLayer(nn.Module, ABC):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_