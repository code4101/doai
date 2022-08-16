from tqdm import tqdm
import numpy as np
import pandas as pd

import paddle
from paddle.io import DataLoader
from paddle.vision import datasets

from pyxllib.xl import TicToc

# 零 配置表
paddle.set_device('gpu')  # 在哪个设备运行，支持写法：cpu、gpu、gpu:0、gpu:1、...
NUM_CLASSES = 10  # 几分类任务
EPOCHS = 5  # 训练轮次
BASE_LR = 0.01  # 学习率
IMS_PER_BATCH = 256  # BATCH_SIZE
STATE_FILE = 'MobileNetV2_model.pdparams'  # 计划存储权重文件的路径


# 一 数据集
def mnist_transform(x):
    # 这里获得的是PIL格式的图片
    y = x.resize([32, 32]).convert('RGB')
    img = np.array(y, dtype='float32') / 255.
    img = img.transpose([2, 0, 1])
    return paddle.to_tensor(img)


# 训练集数量 60000, 28*28 -> 32*32
train_dataset = datasets.MNIST(mode='train', transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=IMS_PER_BATCH)
# 验证集数量 10000, 28*28 -> 32*32
val_dataset = datasets.MNIST(mode='test', transform=mnist_transform)
val_loader = DataLoader(val_dataset, batch_size=IMS_PER_BATCH)

# 数据默认下载到 ~/.cache/paddle/dataset 里

# 二 模型
# 这里特地挑了轻量的MobileNetV2来测试，换成LeNet、resnet18等一样可以运行的。
from paddle.vision.models import MobileNetV2


# 三 训练、推断
def train():
    # 1 准备工作
    model = MobileNetV2(num_classes=NUM_CLASSES)
    optimizer = paddle.optimizer.Adam(learning_rate=BASE_LR, parameters=model.parameters())  # 学习器

    # 2 开始训练
    for epoch in range(EPOCHS):
        loader = DataLoader(train_dataset, batch_size=IMS_PER_BATCH, shuffle=True)
        # f'{epoch=}' 是py3.8语法，低版本可以改成f'epoch={epoch}'，py3.5及以下可以改成 'epoch={}'.fromat(epoch)
        for batch_inputs in tqdm(loader, total=len(train_dataset) // IMS_PER_BATCH, desc=f'{epoch=}'):
            x, y = batch_inputs
            y_hat = model(x)
            loss = paddle.nn.functional.cross_entropy(y_hat, y)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

    paddle.save(model.state_dict(), STATE_FILE)


def eval():
    # 1 加载模型
    model = MobileNetV2(num_classes=NUM_CLASSES)  # 初始化模型结构
    model.set_state_dict(paddle.load(STATE_FILE))  # 加载模型权重
    model.eval()  # 进入推断模式

    # 2 验证集
    with paddle.no_grad():
        gt, pred = [], []
        for batched_inputs in val_loader:
            x, y = batched_inputs
            y_hat = model(x).argmax(axis=1)
            gt += y.reshape([-1]).tolist()
            pred += y_hat.tolist()

    df = pd.DataFrame.from_dict({'gt': gt, 'pred': pred})
    print('验证集各类别出现次数（行ground truth，列pred）')
    print(pd.crosstab(df['gt'], df['pred']))

    correct = sum(df['gt'] == df['pred'])
    total = len(df)
    print(f'正确率: {correct} / {total} ≈ {correct / total:.2%}')


if __name__ == '__main__':
    with TicToc(__name__):
        train()  # 运行消耗显存 1875 MB
        eval()
    # 验证集各类别出现次数（行ground truth，列pred）
    # pred    0     1     2    3    4    5    6     7    8    9
    # gt
    # 0     974     1     0    0    1    1    2     1    0    0
    # 1       0  1127     0    1    1    3    2     0    1    0
    # 2       0     4  1022    0    0    0    1     3    2    0
    # 3       0     0     1  999    0    1    0     6    3    0
    # 4       0     0     0    0  973    0    5     2    0    2
    # 5       2     0     0    5    0  884    0     1    0    0
    # 6       4     2     0    0    1    6  944     0    1    0
    # 7       0     8     3    0    0    0    0  1016    0    1
    # 8       0     0     6    0    0    1    0     2  965    0
    # 9       1     2     2    1   14    0    0     6   12  971
    # 正确率: 9875 / 10000 ≈ 98.75%
    # 2022-08-08 20:22:07 __main__ finished in 1 minute and 38.85 seconds.
