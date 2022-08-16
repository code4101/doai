""" pytorch做分类任务的基本框架（加强版） """

from tqdm import tqdm  # pip install tqdm，进度条工具
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from pyxllib.xl import TicToc  # pip install TicToc，就一个地方用到了，不想安装的删掉TicToc就行


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
        # + transforms，数据预处理，转np.ndarray或torch.tensor结构
        # + data augment，数据增强
        return torch.FloatTensor(x), y


train_loader = DataLoader(NumDataset(5000), batch_size=16)
val_loader = DataLoader(NumDataset(1000), batch_size=16)


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
        """ batched_inputs 总是输入 [x, y] 的结构
        eval阶段可能没有y，eval阶段也用不到y，可以输入[x, None]
        """
        device = next(self.parameters()).device

        x = batched_inputs[0].to(device)
        logits = self.classifier(x)

        if self.training:
            y = batched_inputs[1].to(device)
            loss = nn.functional.cross_entropy(logits, y)
            return loss
        else:
            y_hat = logits.argmax(dim=1)
            return y_hat


def train(epochs=10):
    # 1 加载模型
    model = NumNet()
    model.to('cuda')  # 在哪个设备运行：默认是'cpu'，其他还有 'cuda' 'cuda:0'，'cuda:1'，'cuda:2',...
    optimizer = optim.Adam(model.parameters())

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
    model = NumNet()  # 定义模型结构
    model.load_state_dict(torch.load('model.pth'))  # 加载模型权重
    model.eval()  # 进入推断模式

    # 2 训练集
    with torch.no_grad():
        correct_num = 0
        for batched_inputs in train_loader:
            y_hat = model(batched_inputs)
            correct_num += sum(batched_inputs[1] == y_hat)
        print(f'训练集正确率 {correct_num} / {len(train_loader.dataset)} ≈ {correct_num / len(train_loader.dataset):.2%}')

    # 3 验证集
    with torch.no_grad():
        gt, pred = [], []
        for batched_inputs in val_loader:
            y_hat = model(batched_inputs)
            gt += batched_inputs[1].tolist()
            pred += y_hat.tolist()

    df = pd.DataFrame.from_dict({'gt': gt, 'pred': pred})
    print('验证集各类别出现次数（行ground truth，列pred）：')
    print(pd.crosstab(df['gt'], df['pred']))

    correct = sum(df['gt'] == df['pred'])
    total = len(df)
    print(f'正确率: {correct} / {total} ≈ {correct / total:.2%}')


if __name__ == '__main__':
    with TicToc(__name__):
        train()
        eval()
    # 2021-07-16 07:10:30 time.process_time(): 1.5 seconds.
    # epoch: 100%|████████████████████████████████████| 10/10 [00:11<00:00,  1.17s/it]
    # 训练集正确率 4745 / 5000 ≈ 94.90%
    # 验证集各类别出现次数（行ground truth，列pred）：
    # pred    0    1    2
    # gt
    # 0     382   13    0
    # 1       5  263   25
    # 2       0   11  301
    # 正确率: 946 / 1000 ≈ 94.60%
    # 2021-07-16 07:10:50 __main__ finished in 19.88 seconds.