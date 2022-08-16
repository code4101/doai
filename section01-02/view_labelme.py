#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/09/13 11:05

from pyxllib.xl import TicToc
from pyxllib.data.coco import CocoGtData


def view_labelme():
    """ coco数据格式的可视化

    大概了解下即可，以后数据专题再细讲
    """
    gt_path = r'D:\home\datasets\textGroup\PubLayNet\publaynet\val_mini.json'
    imgs_path = r'D:\home\datasets\textGroup\PubLayNet\publaynet\val_mini'

    cgd = CocoGtData(gt_path)  # 用gt初始化
    cgd.to_labelme(imgs_path)  # 转labelme数据格式，写入json标注文件


if __name__ == '__main__':
    with TicToc(__name__):
        view_labelme()
