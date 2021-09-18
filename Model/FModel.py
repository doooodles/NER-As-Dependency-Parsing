from abc import ABC

from transformers import XLNetModel, BertModel

from Layers import feedforwardLayer, biaffineLayer
import torch.nn as nn
import torch.nn.functional as F
import torch


class FModel(nn.Module, ABC):

    def __init__(self, d_in, d_hid, d_class, n_layers, bi=True, dropout=0.3):
        super().__init__()
        # self.model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
        # self.model_path = r'/data/mgliu/transformers_model/chinese-xlnet-mid'
        # self.model_path = r'/data/mgliu/transformers_model/roberta_chinese_clue_large'
        self.model = BertModel.from_pretrained("clue/roberta_chinese_clue_large")
        # self.m