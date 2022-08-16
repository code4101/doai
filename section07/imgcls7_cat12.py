from pyxllib.prog.pupil import check_install_package

check_install_package('albumentations')

import os
import random

import fire
import numpy as np
from tqdm import tqdm

import paddle
from paddle.vision.models import resnet18

from pyxllib.xl import TicToc, XlPath
from pyxllib.xlcv import xlcv
import pyxlpr.ai.xlpaddle as xlpaddle
from pyxlpr.xlai import XlModel, ImageClasPredictor

paddle.set_device('gpu:0')
os.chdir('/home/chenkunze/data/cat_12')  # 切换到数据所在目录


class ImageClasDataset(xlpaddle.ImageClasDataset):
    @classmethod
    def refine_image_file(cls):
        """ 官方给的图片有点问题，可以跑一下这个函数对图片进行优化

        有的图片，如tO6cKGH8uPEayzmeZJ51Fdr2Tx3fBYSn.jpg格式其实是gif，会导致cv2.imread时报错
        可以用我这里的功能，减小图片体积的同时，可以转换为正确的jpg格式
        """
        from pyxllib.xlcv import ImagesDir

        ImagesDir('cat_12_test').reduce_image_filesize(suffix='.jpg', read_flags=1)
        ImagesDir('cat_12_train').reduce_image_filesize(suffix='.jpg', read_flags=1)

    @classmethod
    def img_augment(cls, img):
        """ 一般都重写这个数据增广的方法

        TODO 这个增广和transform当初是针对另一个ElecLabel配置的，不一定具备泛用性，需要针对这个数据做更多实验
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
        """ 一般要重写这个图片读取的方法 """
        import paddle.vision.transforms.functional as F
        img = xlcv.read(x, 1)
        img = F.resize(img, (256, 256))  # 将图片尺寸统一，方便按batch训练。但resnet并不强制输入图片尺寸大小。
        img = np.array(img, dtype='float32') / 255.
        img = img.transpose([2, 0, 1])
        return img


def train(total_epoch=5, *, save_dir='models/resnet18', batch_size=4):
    train_dataset = ImageClasDataset.from_label('data/train_list.txt', use_img_augment=True)  # 训练集
    model = XlModel(resnet18(pretrained=True, num_classes=train_dataset.num_classes))  # 设置使用的模型
    model.set_save_dir(save_dir)  # 设置保存路径
    model.set_dataset(train_dataset)  # 关联数据集
    model.try_load_params('final.pdparams')  # 尝试读取之前已有训练的权重
    model.prepare_clas_task()  # 分类任务的准备工作。可以自定义optimizer等传进来。
    model.train(total_epoch, batch_size)  # 启动训练与评估


def pred_test_dataset(data_dir='data/cat_12_test'):
    """ 输入数据所在目录，测试运行速度和精度 """
    clas = ImageClasPredictor.from_modeldir('models/resnet18',
                                            dynamic_net=resnet18,
                                            transform=ImageClasDataset.transform)

    res = []
    files = list(XlPath(data_dir).rglob_images('*'))
    for f in tqdm(files):
        cls_id = clas.pred(f)
        res.append(f'{f.name},{cls_id}')
    XlPath('result.csv').write_text('\n'.join(res))


if __name__ == '__main__':
    with TicToc():
        fire.Fire()
