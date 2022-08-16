import numpy as np

import paddle
from paddle.metric import Accuracy
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import Adam
from paddle.vision.datasets import MNIST
from paddle.vision.models import MobileNetV2

from pyxllib.xl import TicToc

# 设备
paddle.set_device('gpu:0')


# 数据

def mnist_transform(x):
    # 这里获得的是PIL格式的图片，如果需要可以通过paddle.vision.set_image_backend('cv2')修改
    y = x.resize([32, 32]).convert('RGB')
    img = np.array(y, dtype='float32') / 255.
    img = img.transpose([2, 0, 1])
    return paddle.to_tensor(img)


train_dataset = MNIST(mode='train', transform=mnist_transform)
val_dataset = MNIST(mode='test', transform=mnist_transform)


def train():
    # 调用MobileNetV2模型，将其传给paddle.Model高层API接口类
    model = paddle.Model(MobileNetV2(num_classes=10))

    # 进行训练前准备
    model.prepare(Adam(0.01, parameters=model.parameters()),  # 优化器
                  CrossEntropyLoss(),  # 损失函数
                  Accuracy()  # 测评函数
                  )

    # 3 启动训练，运行需要近2G显存
    model.fit(train_dataset, val_dataset,  # 训练集、验证集
              epochs=5,
              batch_size=256,
              save_dir='./output'
              )


if __name__ == '__main__':
    with TicToc(__name__):
        train()  # 运行消耗显存 1855 MB
