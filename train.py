
from tqdm import trange
import os
from Model.SModel import SModel
from Model.SNERModel import SNERModel
from utils import MyDataset, BucketDataLoader
from utils.util import batch_computeF1, get_useful_ones
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, XLNetModel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch = 100
batch_size = 12
dataset = MyDataset(path="./utils/train/", count=2515)
trainLoader = BucketDataLoader(dataset, batch_size, True,True)
devLoader = BucketDataLoader(dataset, batch_size, True,False)

model = SNERModel(d_in=768, d_hid=1024, d_class=len(dataset.cateDict) + 1, n_layers=4, dropout=0.5)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
lossFunc = nn.CrossEntropyLoss(reduction='mean')

#model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
pretrained_model = XLNetModel.from_pretrained(dataset.model_path, mem_len=768).to(device).eval()
ckp_path = os.path.join(os.getcwd(), 'checkpoint')

def timeSince(start_time):
    sec = time.time() - start_time
    min = sec//60
    sec = sec%60
    return "{} min {} sec".format(int(min), int(sec))

def evalTrainer():
    epochLoss = 0.0
    cycle = 0
    f1_score = 0
    model.eval()
    for passage, mask, label in devLoader:
        passage = passage.long()
        passage = passage.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if (len(passage.shape) < 2):
            passage = passage.unsqueeze(0)
            mask = mask.unsqueeze(0)

        with torch.no_grad():
            emb = pretrained_model(passage, attention_mask=mask)[0]
            #emb = emb.to(device)

            out = model(emb)
            tmp_out, tmp_label = get_useful_ones(out, label, mask)
            # loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
            #             label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
            loss = lossFunc(tmp_out, tmp_label)

        #score, pred = out.max(-1)
        #print(pred.sum())
        # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()