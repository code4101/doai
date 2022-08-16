import numpy as np

import paddle
from paddle.vision.models import MobileNetV2

from pyxllib.xl import TicToc, dprint
from pyxllib.xlcv import xlpil
from pyxlpr.ai.xlpaddle import ImageClasPredictor  # pip install pyxllib>=0.2.53

paddle.set_device('cpu')


def infer():
    """ 源码部署阶段开发演示
    """

    # 1 初始化预测器
    def mnist_transform(x):
        x = xlpil.read(x)  # 部署阶段，多扩展一个从不同来源，读取为pil图片的功能
        y = x.resize([32, 32]).convert('RGB')
        img = np.array(y, dtype='float32') / 255.
        img = img.transpose([2, 0, 1])
        return paddle.to_tensor(img)

    numcls = ImageClasPredictor.from_dynamic(MobileNetV2(num_classes=10), 'output/final.pdparams',
                                             transform=mnist_transform)

    # 2 使用函数接口功能，执行下游任务
    # 2.1 这是验证集里的图片，训练集和验证集的数据精度都很高，预测基本没有问题
    v1 = numcls.pred('test/000031.jpg')
    dprint(v1)
    # v1<int>=3，准确返回3

    # 2.2 这里我们自己手写几个数字试试，存成文件abcde，这些图片接近部署真实场景，尺寸都是随意的，没有固定到32*32
    v2 = numcls.pred('test/a.jpg')
    dprint(v2)
    # v<int>=1，准确识别为1

    # 2.3 支持批量识别
    vals1 = numcls(['test/b.jpg', 'test/c.jpg', 'test/d.jpg'])
    dprint(vals1)
    # vals1<list>=[2, 3, 4]，b、d、e都正确识别

    # 2.4 传入numpy数据也行
    import cv2
    imgs = [cv2.imread('test/d.jpg'), cv2.imread('test/e.jpg', 0)]
    vals2 = numcls(imgs)
    dprint(vals2)
    # vals2<list>=[4, 5]

    # 2.5 也支持PIL数据格式
    from PIL import Image
    img = Image.open('test/e.jpg')
    vals3 = numcls([img])
    dprint(vals3)
    # vals3<int>=[5]


if __name__ == '__main__':
    with TicToc(__name__):
        infer()
