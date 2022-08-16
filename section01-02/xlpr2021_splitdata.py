#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/03/28

""" 本代码是XLPR 2021级研究生编程考核题的参考答案

【教学】每个脚本的第一个三引号字符串，可以对脚本进行一个整体的介绍
"""

import json
import random


def select_annotations(annotations, image_ids):
    """ 筛选出对应图片的annotations

    【关于函数注释】
    1、py的函数标准注释方法，要写在内部开头三引号字符串里
    2、并建议第一行写功能摘要
    3、:param这种字眼是pycharm自动生成的标准注释结构
        有些特别简单的函数，也没必要非把每个参数、return都解释一遍
        注释本身也有维护成本，有时候代码改了，但注释没改，会给其他使用者带来更大的迷惑性和误导

    【关于封装】
    因为这段“很简单”，本函数可以选择不封装，但封装后代码可读性会略强些。
        但写代码还是要注意不要“过度封装”，何谓过度封装一两句话讲不清，
        要自己多写代码、多学习别人、开源的优秀代码去感悟。
    本函数也可以选择直接嵌套在 split_coco_data 里，对外隐藏，py支持在函数里定义函数

    【doctest】
    这里是doctest文档测试，不仅是使用示例，也是一个测试点。
    这个功能只有按规范在三引号字符串里写才有。
    >>> annotations = [{'image_id': 1, 'box_id': 1}, {'image_id': 1, 'box_id': 2}, {'image_id': 2, 'box_id': 3}]
    >>> select_annotations(annotations, {1, 3})
    [{'image_id': 1, 'box_id': 1}, {'image_id': 1, 'box_id': 2}]
    """
    # 简单的for循环和if操作，可以用“列表推导式”写
    return [an for an in annotations if (an['image_id'] in image_ids)]


def split_coco_data(file, parts, *, shuffle=True):
    """ 数据拆分器，这个数据是coco格式，所以称为coco_data

    :param file: 数据文件，用json格式存储着coco格式的字典
        含有images、annotations、categories三个字段
        本参数也可以设计为传入data字典 （这段看不懂没关系，可以假装不存在）
            但由于random.shuffle是in-place原地操作，
            为了避免修改原始数据，或者要多一步繁琐的copy操作，把读文件也封装了。
            这样这个函数就是完全独立的一个功能组件，不会对外部产生任何难以预知的影响。
    :param dict parts: 每个部分要拆分、写入的文件名，以及数据比例
        py≥3.6的版本中，dict的key是有序的，会按顺序处理开发者输入的清单
        这里比例求和可以不满1，但不能超过1
    :param bool shuffle: 是否打乱原有images顺序
    """
    # 1 读入data  （使用编号注释对代码逻辑分块，能大大提高可读性）
    with open(file, 'r') as f:
        data = json.load(f)
    if shuffle:  # 是否打乱应该设计为一个可选参数，可能会存在某些特殊场合需要维持原数据顺序
        random.shuffle(data['images'])
    assert sum(parts.values()) <= 1, '比例和不能超过1'  # 为了工程鲁棒，常用assert进行简单的规范检查

    # 2 生成每一个部分的文件
    total_num, used_rate = len(data['images']), 0  # 注意py可以这样在一行赋值多个变量 （这个本质是“解包”操作）
    for k, v in parts.items():
        # 2.1 选择子集图片  （代码块出现嵌套逻辑，可以用2.1、2.2的子编号格式来表达）
        images = data['images'][int(used_rate * total_num):int((used_rate + v) * total_num)]

        # 注意这里存成了哈希索引的set集合类型，提高检索效率，算法整体复杂度只有O(n)
        #   如果set是树结构二分查找，复杂度是O(n*log n)
        #   有人用双循环检索的话，复杂度是O(n^2)，速度就很慢
        #   怎么方便、定量、严谨分析算法性能，可以参考前面讲的pyxllib库的TicToc、PerfTest工具
        image_ids = {im['id'] for im in images}  # 改成花括号，列表推导式也能直接生成set类型

        # 2.2 生成新的字典
        part_data = {'images': images,
                     'annotations': select_annotations(data['annotations'], image_ids),
                     'categories': data['categories']}

        # 2.3 写入文件
        # f字符串：f'{k}.json' 等价于 str(k) + '.json'
        with open(f'{k}.json', 'w', encoding='utf8') as f:
            json.dump(part_data, f)

        # 2.4 更新使用率
        used_rate += v


if __name__ == '__main__':
    # 不要把执行代码放在外面，养成好习惯放在if __name__ == '__main__'里面，并对有一定规模的代码都封装成函数
    split_coco_data('publaynet.json', {'train': 0.6, 'val': 0.2, 'test': 0.2})
