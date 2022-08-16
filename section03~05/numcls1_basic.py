""" pytorch做分类任务的基本框架（基础版） """

from tqdm import tqdm  # pip install tqdm，进度条工具
import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader


class NumDataset(Dataset):
    def __init__(self, n):
        self.data = np.random.randint(0, 6, [n, 5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        s = sum(x)

        if s < 12:
            y = 0
        elif s < 15:
            y = 1
        else:
            y = 2

        # + transforms，数据预处理
        # + data augment，数据增强
        return torch.FloatTensor(x), y


train_loader = DataLoader(NumDataset(5000), batch_size=16, shuffle=True)
val_loader = DataLoader(NumDataset(1000), batch_size=16, shuffle=True)


class NumNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 模型结构
        self.classifier = nn.Sequential(
            nn.Linear(in_features=5, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=5),
            nn.LeakyReLU(),
            nn.Linear(in_features=5, out_features=3),
            nn.Sigmoid(),
        )

        # 2 初始化权重
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, batched_inputs):
        device = next(self.parameters()).device

        x, y = batched_inputs
        x = x.to(device)
        logits = self.classifier(x)

        if self.training:
            y = y.to(device)
            loss = nn.functional.cross_entropy(logits, y)
            return loss
        else:
            y_hat = logits.argmax(dim=1)
            return y_hat


def train(epochs=10):
    # 1 加载模型
    model = NumNet()
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 2 训练
    for epoch in tqdm(range(epochs), 'epoch'):
        for batched_inputs in train_loader:
            loss = model(batched_inputs)  # 前向传播
            optimizer.zero_grad()  # 清空之前梯度
            loss.backward()  # 进行该轮梯度反传
            optimizer.step()  # 按指定学习策略，更新网络权重的梯度

    # 3 保存
    torch.save(model.state_dict(), 'model.pth')


def eval():
    # 1 加载模型
    model = NumNet()  # 初始化模型结构
    model.load_state_dict(torch.load('model.pth'))  # 加载模型权重
    model.eval()  # 进入推断模式

    # 2 检查分类精度
    with torch.no_grad():
        correct_num = 0
        for batched_inputs in train_loader:
            y_hat = model(batched_inputs)
            correct_num += sum(batched_inputs[1] == y_hat)
    print(f'正确率: {correct_num} / {len(train_loader.dataset)} ≈ {correct_num / len(train_loader.dataset):.2%}')

    # 3 验证集
    with torch.no_grad():
        gt, pred = [], []
        for batched_inputs in val_loader:
            y_hat = model(batched_inputs)
            gt += batched_inputs[1].tolist()
            pred += y_hat.tolist()

    df = pd.DataFrame.from_dict({'gt': gt, 'pred': pred})
    print('验证集各类别出现次数（行ground truth，列pred）')
    print(pd.crosstab(df['gt'], df['pred']))

    correct = sum(df['gt'] == df['pred'])
    total = len(df)
    print(f'正确率: {correct} / {total} ≈ {correct / total:.2%}')


if __name__ == '__main__':
    np.random.seed(4102)
    torch.manual_seed(4102)  # torch也有随机数种子

    train()
    eval()
