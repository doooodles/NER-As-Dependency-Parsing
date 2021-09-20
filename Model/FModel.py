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
        # self.model = BertModel.from_pretrained(self.model_path)
        # self.model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=1024)
        self.bilstm = nn.LSTM(d_in, 200, num_layers=3, batch_first=True, dropout=0.4,
                              bidirectional=bi)
        self.feedStart = feedforwardLayer(400, 150, dropout=0.2)
        self.feedEnd = 