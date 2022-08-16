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

EVAL_PERIOD = 200

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
    from visdom import Visdom  # 如果在开头写，这里就不用import了

    model = LeNet5(NUM_CLASSES)
    model.to(DEVICE)  # 使用cpu，还是哪个gpu训练
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)  # 学习器

    vis = Visdom()
    vis.close('loss')
    vis.close('eval')

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
        # 2.2.1 按周期保存模型，防止突然断网、报错，需要重新开始训练
        if CHECKPOINT_PERIOD and i % CHECKPOINT_PERIOD == 0:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            torch.save(model.state_dict(), str(STATE_FILE))

        # 2.2.2 loss可视化
        if vis.win_exists('loss'):  # 添加数据
            vis.line([float(loss)], [i], 'loss', update='append')
        else:  # 第一次展示窗口
            vis.line([float(loss)], [i], 'loss', opts={'title': 'loss', 'xlabel': 'epoch'})

        # 2.2.3 模型精度中间结果可视化（含train和test数据效果对比）
        if EVAL_PERIOD and i % EVAL_PERIOD == 0:
            predictor = XlPredictor(model)
            train_f1 = ClasEvaluater(predictor.forward(train_loader)).f1_score()
            test_f1 = ClasEvaluater(predictor.forward(val_loader)).f1_score()
            model.train()  # 计算完要主动转回train模式

            if vis.win_exists('eval'):
                vis.line([[train_f1, test_f1]], [i], 'eval', update='append')
            else:
                vis.line([[train_f1, test_f1]], [i], 'eval',
                         opts={'title': '模型精度', 'legend': ['train', 'test']})


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
