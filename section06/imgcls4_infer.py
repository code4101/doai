from torchvision import transforms

from pyxllib.xl import XlPath, TicToc, dprint  # pip install pyxllib
from pyxllib.xlcv import xlcv  # pip install pyxllib[xlcv]
from pyxlpr.ai.torch import LeNet5, XlPredictor  # 封装了一些通用组件，简化开发

NUM_CLASSES = 10  # 几分类任务
IMS_PER_BATCH = 200  # BATCH_SIZE
STATE_FILE = XlPath('mnist/lenet5_model.pth')  # 计划存储权重文件的路径


def infer():
    """ 源码部署阶段开发演示
    """
    # 1 初始化预测器：因为是预测真实数据，没有y标签，可以y_placeholder=-1作为占位符制作dataset数据集
    #   权重文件支持给url，会自动下载到本地，在部署一些小模型、可公开功能的时候很方便。也方便在云端替换最新版最好的权重文件。
    #   如果读者写的model.forward前传机制不同，本来batch_inputs就只输入x没有y，则这里不用设置y_placeholder参数
    numcls = XlPredictor(LeNet5(NUM_CLASSES), STATE_FILE, 'cuda', batch_size=IMS_PER_BATCH, y_placeholder=-1)
    numcls.transform = transforms.Compose([
        lambda x: xlcv.read(x, 0),  # 我自己的一个类，能自动读取文件、或者转换格式成numpy数据，类似cv2.imread，但比其强大的多
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])

    # 2 使用函数接口功能，执行下游任务
    # 2.1 这是验证集里的图片，训练集和验证集的数据精度都很高，预测基本没有问题
    v1 = numcls('test/000031.jpg')
    dprint(v1)
    # v1<int>=3，准确返回3

    # 2.2 这里我们自己手写几个数字试试，存成文件abcde，这些图片接近部署真实场景，尺寸都是随意的，没有固定到32*32
    v2 = numcls('test/a.jpg')
    dprint(v2)
    # v<int>=1，准确识别为1

    # 2.3 支持批量识别，forward的时候会使用前面设置的batch_size批量前传
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
