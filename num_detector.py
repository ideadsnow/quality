# /usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import cv2
import numpy
import math
import os
import csv
import logging
# numpy.set_printoptions(threshold=numpy.nan)

# logging.basicConfig(filename='detector.log', level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def GaussianBlur(img, kernel_size, sigma):
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    # cv2.imshow('s', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


# 中点坐标
CENTER_POS = (
    (1.5, 1.5),
    (0.75, 3),
    (2.25, 3),
    (1.5, 4.5),
    (0.75, 6),
    (2.25, 6),
    (1.5, 7.5)
)
# 0～9 每个数字所对应七个长条的关系（0 表示不亮，1 表示亮）
BITMAP = {
    (1, 1, 1, 0, 1, 1, 1): '0',
    (0, 0, 1, 0, 0, 1, 0): '1',
    (1, 0, 1, 1, 1, 0, 1): '2',
    (1, 0, 1, 1, 0, 1, 1): '3',
    (0, 1, 1, 1, 0, 1, 0): '4',
    (1, 1, 0, 1, 0, 1, 1): '5',
    (1, 1, 0, 1, 1, 1, 1): '6',
    (1, 0, 1, 0, 0, 1, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9'
}


# 判断图像中某点及其上下左右四个点是否有白色
def is_white(img, dot):
    # cv2.imwrite('num_block.bmp', img)
    logger.debug('dot: %s' % str(dot))
    # exit()
    y = int(math.ceil(dot[1]))
    x = int(math.ceil(dot[0]))
    pixel = img[y:y + 1, x:x + 1][0]
    pixel_left = img[y:y + 1, x - 1:x][0]
    pixel_right = img[y:y + 1, x + 1:x + 2][0]
    pixel_top = img[y - 1:y, x][0]
    pixel_bottom = img[y + 1:y + 2, x][0]
    logger.debug('pixel: %s' % pixel)
    white_flag = 1 if [255] in [pixel_left, pixel_right,
                                pixel_top, pixel_bottom, pixel] else 0
    return white_flag


# 识别单个数字
def _num_detect(block, h, w):
    bit_map = [0, 0, 0, 0, 0, 0, 0]
    for i in range(7):
        bit_map[i] = is_white(
            block, (CENTER_POS[i][0] * w, CENTER_POS[i][1] * h))
        # cv2.imshow('s', block)
        # cv2.waitKey()
        # print bit_map
        logger.debug('bit_map: %s' % bit_map)
        # exit()
    num = BITMAP.get(tuple(bit_map), '')
    logger.debug('bit_map: %s' % bit_map)
    return num


# 无外框的数字块区域，目前使用的视频 h=10，zfill=4
def num_detect(num_block, zfill=4):
    # 将 num_block 按照数字位数 zfill 值分开
    # cv2.imshow('s', num_block)
    # cv2.imwrite('num_block.bmp', num_block)
    # cv2.waitKey(0)
    # exit()
    length = int(num_block.shape[1] / zfill)
    h = num_block.shape[0] / 9
    w = length / 3
    blocks = []
    for i in range(zfill):
        blocks.append(num_block[:, length * i:length * (i + 1)])
    num_str = ''
    for block in blocks:
        # cv2.imshow('s', block)
        # cv2.waitKey()
        # cv2.imwrite('block.bmp', block)
        num = _num_detect(block, h, w)
        if num == '':
            # TODO 错误处理
            return ''
        num_str = num_str + num
    return num_str


def is_white2black(prev, cur):
    return prev == 255 and cur == 0


def is_black2white(prev, cur):
    return prev == 0 and cur == 255


def detect_num_block(img):
    # 高斯模糊
    img_blur = GaussianBlur(img, (3, 3), 3)
    # 灰度图
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # 二值化
    border_thresh, img_bin = cv2.threshold(
        img_gray, 230, 255, cv2.THRESH_BINARY)
    # cv2.imshow('s', img_bin)
    # cv2.waitKey(0)
    # cv2.imwrite('img_bin.bmp', img_bin)
    # exit()

    # TODO 补全注释，搜索算法
    start_x = int(img_bin.shape[1] / 8 * 3)
    start_y = int(img_bin.shape[0] / 4 * 3)
    prev = -1
    cur = -1
    lt = [-1, -1]

    x = start_x
    y = start_y

    # 找白框左边界
    while y < img_bin.shape[0] and lt[0] == -1:
        y += 5
        x = start_x
        while x < img_bin.shape[1] and lt[0] == -1:
            prev = cur
            cur = img_bin[y][x]
            if is_black2white(prev, cur):
                lt[0] = x
            x += 1

    if lt[0] == -1:
        logger.error('left side did not found.')
        return

    # 找白框左上角顶点
    prev = -1
    cur = -1
    while y > start_y and lt[1] == -1:
        y -= 1
        prev = cur
        cur = img_bin[y][x]
        if is_white2black(prev, cur):
            lt[1] = y + 1

    # 从白框左上角出发，逐行扫描，找到数字块左上角的坐标
    num_block_lt = [-1, -1]
    # 目前原图白框为 20px，直接跳过
    x, y = lt[0], lt[1] + 20 * img_bin.shape[0] / 1080
    prev = -1
    cur = -1
    while num_block_lt[0] == -1:
        if y >= img_bin.shape[0]:
            break
        y += 1
        x = lt[0]
        while num_block_lt[0] == -1:
            if x >= img_bin.shape[1]:
                break
            prev = cur
            cur = img_bin[y][x]
            if is_white2black(prev, cur):
                num_block_lt[0] = x
            x += 1
    if num_block_lt[0] == -1:
        logger.error('left side did not found.')
        return
    while y > lt[1] and num_block_lt[1] == -1:
        if y >= img_bin.shape[0]:
            break
        y -= 1
        prev = cur
        cur = img_bin[y][x]
        if is_black2white(prev, cur):
            num_block_lt[1] = y + 1
    if num_block_lt[1] == -1:
        logger.error('left top did not found.')
        return
    # print num_block_lt
    # 从数字块左上角出发，找到数字块下边界和右边界
    num_block_rb = [-1, -1]
    x, y = num_block_lt
    prev = -1
    cur = -1
    while num_block_rb[1] == -1:
        if y >= img_bin.shape[0]:
            break
        y += 1
        prev = cur
        cur = img_bin[y][num_block_lt[0]]
        if is_black2white(prev, cur):
            num_block_rb[1] = y - 1
    if num_block_rb == -1:
        logger.error('bottom side did not found.')
        return
    while num_block_rb[0] == -1:
        if x >= img_bin.shape[1]:
            break
        x += 1
        prev = cur
        cur = img_bin[num_block_lt[1]][x]
        if is_black2white(prev, cur):
            num_block_rb[0] = x - 1
    if num_block_rb[0] == -1:
        logger.error('right bottom did not found.')
        return
    # print num_block_rb

    num_block_info = (min_x, min_y, max_x,
                      max_y) = num_block_lt[0], num_block_lt[1], num_block_rb[0], num_block_rb[1]

    return num_block_info


def detect(img, block_info):
    min_x, min_y, max_x, max_y = block_info

    # 高斯模糊
    img_blur = GaussianBlur(img, (3, 3), 3)
    # 灰度图
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # 二值化
    border_thresh, img_bin = cv2.threshold(
        img_gray, 127, 255, cv2.THRESH_BINARY)

    num_block = img_bin[min_y:max_y + 1, min_x:max_x + 1]

    # cv2.imshow('s', num_block)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    num_str = num_detect(num_block)
    return num_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.img_dir)

    # import time
    # start = time.time()

    images = [os.path.join(base_dir, x) for x in os.listdir(
        base_dir) if x.lower().endswith('.bmp')]

    if not images:
        logger.error('no images found.')
        exit()

    # block_info = [839, 899, 1080, 990]
    block_info = detect_num_block(cv2.imread(images[0]))
    if not block_info:
        logger.error('num block detect failed.')
        exit(0)
    # print block_info
    logger.debug('block_info: %s' % block_info)

    with open('detect_result.csv', 'wb') as csvfile:
        for img in images:
            num_str = detect(cv2.imread(img), block_info)
            # if num_str and (int(num_str) < 301 or int(num_str) > 7506):
            # continue
            writer = csv.writer(csvfile)
            writer.writerow([img, '{:0>6}'.format(num_str)])

    # end = time.time()
    # logger.info('total time: %d' % (end - start))


# 单张图片识别测试
def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    args = parser.parse_args()

    img = cv2.imread(args.img)

    block_info = detect_num_block(img)
    if not block_info:
        logger.error('num block detect failed.')
        exit(0)

    num_str = detect(img, block_info)
    logger.info('num str: %s' % num_str)


if __name__ == '__main__':
    # main()
    main2()
