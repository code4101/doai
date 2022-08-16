import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pyxllib.xl import TicToc
from pyxlpr.ai.torch import TrainingSampler, LeNet5, XlPredictor, ClasEvaluater  # 封装了一些通用组件，简化开发

# 零 配置表
DATA_DIR = 'datasets'  # 数据集所在根目录
NUM_CLASSES = 10  # 几分类任务
DEVICE = 'cuda'  # 在哪个设备运行

BASE_LR = 0.01  # 学习率
IMS_PER_BATCH = 200  # BATCH_SIZE
MAX_ITER = 600  # 训练集迭代次数

CHECKPOINT_PERIOD = 100  # 每迭代多少次保存模型
STATE_FILE = 'mnist/lenet5_model.pth'  # 计划存储权重文件的路径

# 一 数据集
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
])
# 训练集数量 60000, 28*28 -> 32*32
train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=IMS_PER_BATCH)
# 验证集数量 10000, 28*28 -> 32*32
val_dataset = datasets.MNIST(DATA_DIR, train=False, transform=mnist_transform)
val_loader = DataLoader(val_dataset, batch_size=IMS_PER_BATCH)


# 二 训练、推断
def train():
    # 1 准备工作
    model = LeNet5(NUM_CLASSES)
    model.to(DEVICE)  # 使用cpu，还是哪个gpu训练
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)  # 学习器

    # 2 开始训练
    # 这里不用前面预设的train_loader，而是新建一个可以无限迭代的loader
    loader = DataLoader(train_dataset, batch_size=IMS_PER_BATCH,
                        sampler=TrainingSampler(len(train_dataset)))
    for i, batch_inputs in tqdm(enumerate(loader, 1), 'train', MAX_ITER):
        # 2.1 训练终止标记/正常训练过程
        if i > MAX_ITER:
            break
        loss = model(batch_inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 2.2 扩展功能，可以根据i进行某些周期性操作，也可以写一些每次迭代都处理的操作
        if CHECKPOINT_PERIOD and i % CHECKPOINT_PERIOD == 0:
            # 确保父级目录存在，也可以用 os.makedirs(str(STATE_FILE.parent), exist_ok=True) 实现
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            torch.save(model.state_dict(), STATE_FILE)


def eval():
    predictor = XlPredictor(LeNet5(NUM_CLASSES), STATE_FILE, DEVICE, batch_size=IMS_PER_BATCH)

    print('【训练集】')  # 训练集一般不做eval。但为了分析过拟合欠拟合问题，是可以对比验证集看一下的。
    # 因为要返回gt给下游的eval计算分值，所以要打开return_gt，这跟部署阶段使用模式是不一样的
    preds = predictor.forward(train_loader, print_mode=True)  # 得到所有数据的预测结果
    evaluater = ClasEvaluater.from_pairs(preds)  # 测评器
    print(evaluater.f1_score('all'))  # 训练集看下总精度就行了

    print('【验证集】')
    preds = predictor.forward(val_loader, print_mode=True)
    evaluater = ClasEvaluater.from_pairs(preds)
    print(evaluater.crosstab())  # 验证集可以详细看下交叉表
    print(evaluater.f1_score('all'))


if __name__ == '__main__':
    with TicToc(__name__):
        train()
        eval()
    # 不设种子seed的话，每次结果都会不太一样，但这个模型效果基本稳定在0.97
    # 2021-07-23 10:51:47 time.process_time(): 2.58 seconds.
    # train: 100%|██████████| 600/600 [00:27<00:00, 21.85it/s]
    # 【训练集】
    # eval batch: 100%|██████████| 300/300 [00:12<00:00, 24.59it/s]
    # {'f1_weighted': 0.9743, 'f1_macro': 0.9743, 'f1_micro': 0.9744}
    # 【验证集】
    # eval batch: 100%|██████████| 50/50 [00:02<00:00, 21.51it/s]
    # pred    0     1    2    3    4    5    6     7    8    9
    # gt
    # 0     957     0    3    0    1    2    4     8    2    3
    # 1       0  1126    0    2    0    2    2     1    2    0
    # 2       6     5  967   27    4    0    1    13    9    0
    # 3       0     0    1  992    0    2    0     8    3    4
    # 4       2     0    1    0  970    0    2     1    0    6
    # 5       4     0    0    9    0  869    3     1    3    3
    # 6       8     2    2    0    6    3  935     0    2    0
    # 7       0     4    1    6    1    0    0  1016    0    0
    # 8       6     4    1   11    4    3    1     7  935    2
    # 9       1     6    0    3   21    3    0    14    4  957
    # {'f1_weighted': 0.9724, 'f1_macro': 0.9723, 'f1_micro': 0.9724}
    # 2021-07-23 10:52:33 __main__ finished in 45.52 seconds.
