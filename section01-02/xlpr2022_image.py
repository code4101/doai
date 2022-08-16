import cv2


def __1_create():
    """ 生成测试数据 """


def get_primes(min_num, max_num):
    """ 比较简单的素数生成方法，更高效的方法参考：
    https://stackoverflow.com/questions/567222/simple-prime-number-generator-in-python

    >>> get_primes(1, 10)
    [2, 3, 5, 7]
    """
    from math import sqrt, ceil

    primes = []
    for i in range(max(min_num, 2), max_num):
        for j in range(2, min(ceil(sqrt(i) + 1), i)):
            if i % j == 0:
                break
        else:
            primes.append(i)
    return primes


def create_image(name, outfile):
    import random
    import PIL.Image
    from pyxllib.xlcv import xlpil

    # 1 抽两个质数作为图片的宽和高
    h, w = random.sample(get_primes(300, 400), 2)
    im = PIL.Image.new('RGB', (h, w))
    # 2 写入文字
    im = xlpil.plot_text(im, (10, 100), f'{name}，', font_size=50)
    im = xlpil.plot_text(im, (10, 200), '恭喜通关！', font_size=50)
    # 3 变换数据
    im = xlpil.to_cv2_image(im)
    im = im.transpose([1, 0, 2])
    im = im.reshape([1, -1, 3])  # 打平
    # 4 保存文件
    cv2.imwrite(outfile, im)


def __2_restore():
    """ 复原图片 """


def primefactors(n):
    """ 生成数值n的所有质因数

    >>> primefactors(15)
    [3, 5]
    """
    nums = []
    f = 2
    while f * f <= n:
        while not n % f:
            nums.append(f)
            n //= f
        f += 1
    if n > 1:
        nums.append(n)
    return nums


def restore_image(infile):
    im = cv2.imread(infile)
    h, w = primefactors(im.shape[1])
    print(h, w)

    im1 = im.reshape([h, w, 3])
    im2 = im.reshape([w, h, 3])
    cv2.imshow('im1', im1)
    cv2.imshow('im2', im2)
    cv2.imshow('im3', cv2.transpose(im1))
    cv2.imshow('im4', cv2.transpose(im2))

    cv2.waitKey()


if __name__ == '__main__':
    # create_image('欢迎师弟师妹', 'chaos.png')
    restore_image('chaos0.png')
