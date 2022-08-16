#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/03/21 15:06

import numpy as np
import torch
from torch.utils.data import Dataset


class NumDataset(Dataset):
    def __init__(self, n):
        self.data = np.random.randint(0, 6, [n, 5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = int(10 < sum(x) < 20)
        # + transforms，数据预处理
        # + data augment，数据增强
        return torch.FloatTensor(x), y
