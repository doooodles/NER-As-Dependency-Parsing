from itertools import chain
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import trange


class dataPreLoader:
    def __init__(self, path: str = "./train/", count: int = 2515):
        self.data, self.index2cate, self.cate2index = self.dataFilter(path, count)

    def loadLabel(self, path: str = "./train/", count: int = 2515) -> list:
        labelList = []
        for i in trange(count):
            # 跳过两个会报错的label文件
            if i == 30 or i == 667:
                continue
            temp = pd.read_csv(path + "label/" + str(i) + ".csv")
            sentence = pd.read_csv(path + "data/" + str(i) + ".txt", names=['Text'])['Text']
            sentence = ''.join(list(sentence))
            temp['Text'] = sentence[:min(len(sentence), 500)]
            temp = temp[temp["Pos_b"] < 500][temp["