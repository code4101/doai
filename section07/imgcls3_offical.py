import paddle
from paddle.vision.models import resnet18
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.transforms import Transpose

# 0 设备
paddle.set_device('gpu:0')

# 1 数据
paddle.vision.set_image_backend('cv2')
train_dataset = Cifar10(mode='train', transform=Transpose())
val_dataset = Cifar10(mode='test', transform=Transpose())

# 2 模型
model = paddle.Model(resnet18(pretrained=False, num_classes=10))

# 进行训练前准备
optimizer = Momentum(learning_rate=0.01,
                     momentum=0.9,
                     weight_decay=L2Decay(1e-4),
                     parameters=model.parameters())
model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))

# 3 启动训练
model.fit(train_dataset,
          val_dataset,
          epochs=50,
          batch_size=64,
          save_dir="./output",
          num_workers=8)
