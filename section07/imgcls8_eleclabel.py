from pyxllib.prog.pupil import check_install_package

check_install_package('albumentations')

import os
import random
import time

import fire
import numpy as np
from tqdm import tqdm

from paddle import nn
import paddle
from paddle.metric import Accuracy
from paddle.optimizer import AdamW
from paddle.vision.models import resnet18

from pyxllib.xl import TicToc, ValuesStat, XlPath
from pyxllib.xlcv import xlcv
import pyxlpr.ai.xlpaddle as xlpaddle
from pyxlpr.xlai import XlModel, ImageClasPredictor

paddle.set_device('gpu:0')
os.chdir('/home/chenkunze/data/ElecLabel')  # 切换到数据所在目录


class ImageClasDataset(xlpaddle.ImageClasDataset):
    @classmethod
    def img_augment(cls, img):
        """ 因为图片数据比较少，这里做个数据增强

        这里会随机改变图片，但不改变类别标签。
        其实如果增加mixup之类的操作会更好，但那个代码组织更麻烦，先不考虑。
        """
        import albumentations as A
        h, w, c = img.shape
        # 如果进行随机裁剪，则h, w的尺寸变化
        h = random.randint(int(h * 0.7), h)
        w = random.randint(int(w * 0.7), w)
        transform = A.Compose([
            A.RandomCrop(width=w, height=h, p=0.8),
            A.CoarseDropout(),  # 随机噪声遮挡
            A.RandomSunFlare(p=0.1),  # 随机强光
            A.RandomShadow(p=0.1),  # 随机阴影
            A.RGBShift(p=0.1),  # RGB波动
            A.Blur(p=0.1),  # 模糊
            A.RandomBrightnessContrast(p=0.2),  # 随机调整图片明暗
        ])
        return transform(image=img)['image']

    @classmethod
    def transform(cls, x):
        import paddle.vision.transforms.functional as F
        img = xlcv.read(x)
        img = F.resize(img, (256, 256))  # 将图片尺寸统一，方便按batch训练。但resnet并不强制输入图片尺寸大小。
        img = np.array(img, dtype='float32') / 255.
        img = img.transpose([2, 0, 1])
        return img

    def __getitem__(self, index):
        file, label = self.samples[index]
        img = xlcv.read(file)
        if self.use_img_augment:
            img = self.img_augment(img)
        img = self.transform(img)
        return img, np.array(label, dtype='int64')


def train(total_epoch=100, *, save_dir='models/clas_resnet18', batch_size=4):
    """ 训练模板分类模型 """
    # 1 数据
    # 如果目录已经分好类别，可以直接输入 data/train、data/val
    # 因为数据量太少，这里不分train和val，已各种增广过的数据作为train，以原始图作为val
    train_dataset = ImageClasDataset.from_folder('data/1模板分类', use_img_augment=True)  # 训练集
    val_dataset = ImageClasDataset.from_folder('data/1模板分类')

    # 2 模型及其他相关配置
    model = XlModel(resnet18(pretrained=True, num_classes=train_dataset.num_classes))  # 设置使用的模型
    model.set_save_dir(save_dir)  # 设置保存路径
    model.set_dataset(train_dataset, val_dataset)  # 关联数据集
    model.try_load_params('final.pdparams')  # 尝试读取之前已有训练的权重
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.005, total_epoch)
    optimizer = AdamW(learning_rate=scheduler, parameters=model.parameters())
    model.prepare_clas_task(optimizer)  # 分类任务的准备工作。可以自定义optimizer等传进来。
    model.train(total_epoch, batch_size)  # 启动训练与评估

    # 3 启动训练与评估
    model.train(total_epoch, batch_size)
    # 如果发现精度没有到100%，可以尝试加大epoch多跑一下

    # 4 导出静态图部署模型（不需要做部署的可以删掉这部分）
    model.save_static_network()


def get_clas(dynamic=False):
    """ 获得分类器

    这里只是为了进行性能测试对比，实际部署的时候，不用考虑动态图，可以直接部署静态图
    """
    if dynamic:
        template_clas = ImageClasPredictor.from_modeldir('models/clas_resnet18',
                                                         dynamic_net=resnet18,
                                                         transform=ImageClasDataset.transform)
    else:
        template_clas = ImageClasPredictor.from_modeldir('models/clas_resnet18',
                                                         transform=ImageClasDataset.transform)
    return template_clas


def demo():
    """ 使用示例 """
    clas1 = get_clas(dynamic=True)
    file = "data/1模板分类/安富利科技香港有限公司/cut__1640055868510_95c62475-4d42-4af2-9501-e4f8e4d35536.jpg"
    print(clas1.pred(file))  # 安富利科技香港有限公司

    clas2 = get_clas()
    print(clas2.pred(file))  # 安富利科技香港有限公司


def test(data_dir='data/1模板分类'):
    """ 输入数据所在目录，测试运行速度和精度 """

    def run_clas(clas):
        # 先识别一张，排除掉初始化时间
        clas.pred(files[0])

        timedata = []
        acc = 0  # 模板识别正确的数量
        for f in tqdm(files):
            start = time.time()
            pred_res = clas.pred(f)
            timedata.append(time.time() - start)
            # 输出模板类别识别错的情况
            if pred_res != f.parent.name:
                pass
                # print(f, '模板识别错->', pred_res)
            else:
                acc += 1

        print(f'模板分类精度 {acc}/{total}≈{acc / total:.2%}')
        print('测速(秒) ' + ValuesStat(timedata).summary(valfmt='.3f'))

    files = list(XlPath(data_dir).rglob_images('*'))
    total = len(files)

    # clas1 = get_clas(dynamic=True)
    clas2 = get_clas()

    # run_clas(clas1)
    # 模板分类精度 162/361≈44.88%
    # 测速(秒) 总和: 14.809	均值标准差: 0.041±0.012	总数: 361	最小值: 0.030	最大值: 0.092

    run_clas(clas2)
    # 模板分类精度 161/361≈44.60%
    # 测速(秒) 总和: 44.872	均值标准差: 0.124±0.011	总数: 361	最小值: 0.113	最大值: 0.182


if __name__ == '__main__':
    with TicToc():
        fire.Fire()
