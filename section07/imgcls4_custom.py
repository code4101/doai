import numpy as np

import paddle
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import Adam
from paddle.vision.datasets import MNIST
from paddle.vision.models import MobileNetV2

from pyxllib.xl import TicToc
from pyxlpr.ai.xlpaddle import ClasAccuracy, VisualAcc  # pip install pyxllib>=0.2.53

# 设备
paddle.set_device('gpu:0')


# 数据

def mnist_transform(x):
    # 这里获得的是PIL格式的图片
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
                  ClasAccuracy(print_mode=2)  # 测评函数，改成了自定义测评类
                  )

    # 启动训练
    model.fit(train_dataset, val_dataset,  # 训练集、验证集
              epochs=1,
              batch_size=256,
              save_dir='./output',
              log_freq=50,
              save_freq=2,  # 注意这个周期计算方式：内部输入的epoch是0、1、2、3、4，然后除以周期取余为0时执行
              # eval_freq=2,  # 这个配合作图，会有些使用体验上的瑕疵。为了可视化简洁些，暂不开启这个参数。
              callbacks=[VisualAcc('./output', 'exp001')]
              )


if __name__ == '__main__':
    with TicToc(__name__):
        train()
