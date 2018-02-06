# /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import math
import logging
import os


class Detector(object):
    """
        1. 从输入原始图像中寻找数字块
        2. 根据指定参数等宽切割数字块得到单个数字块集合
        3. 遍历并识别单个数字块中的数字
        4. 依次拼接数字，返回识别结果

        tips：
            1. 目前使用原始图片白框部分宽度为 20px
            2. 数字区域（黑色背景）宽高分别为：
                W = (k+4)h
                H = (2k+5)h

    """
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

    # 0～9 每个数字所对应七个长条的关系（0 表示黑，1 表示白）
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

    # 数字区域坐标缓存
    NUM_BLOCK_INFO = None

    def __init__(self, img_path='unknown', img=None, zfill_width=6,
                 kernel_size=(3, 3), sigma=3, num_size=4, logger=None):
        if logger is None:
            logging.basicConfig(filename='num_detector.log', level=logging.DEBUG)
            logger = logging.getLogger('num_detector')
        self.logger = logger

        if img_path != 'unknown':
            self.img_path = os.path.abspath(img_path)
            self.img = cv2.imread(self.img_path)
        else:
            self.img_path = 'unknown'
            self.img = img
        if self.img is None:
            raise RuntimeError('read image %s error.' % self.img_path)

        # 高斯模糊参数
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.num_block = None

        self.nums = []

        self.zfill_width = zfill_width
        if type(self.zfill_width) is not int:
            raise TypeError('zfill width must be int but {0} now.'.format(type(self.zfill_width).__name__))

        # 图片中数字块的数字位数
        self.num_size = num_size

        self.run()

    def run(self):
        self.logger.info('%s detecting...' % self.img_path)
        if Detector.NUM_BLOCK_INFO is None:
            self.detect_num_block()

        min_x, min_y, max_x, max_y = Detector.NUM_BLOCK_INFO

        # 高斯模糊
        img_blur = cv2.GaussianBlur(self.img, self.kernel_size, self.sigma)
        # 灰度图
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        # 二值化
        border_thresh, img_bin = cv2.threshold(
            img_gray, 127, 255, cv2.THRESH_BINARY)

        self.num_block = img_bin[min_y:max_y + 1, min_x:max_x + 1]

        # cv2.imshow('s', self.num_block)
        # cv2.waitKey()
        # return

        self.detect()
        self.logger.info('%s detect finish.' % self.img_path)

    # 从输入原图搜索数字块
    def detect_num_block(self):
        img = self.img

        # TODO 把坑填了
        try:
            self.logger.info('detecting num block. %s' % self.img_path)
            # 高斯模糊
            img_blur = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
            # 灰度图
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
            # 二值化
            border_thresh, img_bin = cv2.threshold(
                img_gray, 230, 255, cv2.THRESH_BINARY)
            # cv2.imshow('s', img_bin)
            # cv2.waitKey(0)
            # cv2.imwrite('img_gray.bmp', img_gra)
            # cv2.imwrite('img_bin.bmp', img_bin)
            # exit()

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
                    if self.__is_b2w(prev, cur):
                        lt[0] = x
                    x += 1

            if lt[0] == -1:
                self.logger.error('outer block left side did not found.')
                return
            self.logger.info('outer block left side found.')

            # 找白框左上角顶点
            prev = -1
            cur = -1
            while y > start_y and lt[1] == -1:
                y -= 1
                prev = cur
                cur = img_bin[y][x]
                if self.__is_w2b(prev, cur):
                    lt[1] = y + 1

            self.logger.info('outer block left top point found.')

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
                    if self.__is_w2b(prev, cur):
                        num_block_lt[0] = x
                    x += 1
            if num_block_lt[0] == -1:
                self.logger.error('inner block left side did not found.')
                return
            self.logger.info('inner block left side found.')
            while y > lt[1] and num_block_lt[1] == -1:
                if y >= img_bin.shape[0]:
                    break
                y -= 1
                prev = cur
                cur = img_bin[y][x]
                if self.__is_b2w(prev, cur):
                    num_block_lt[1] = y + 1
            if num_block_lt[1] == -1:
                self.logger.error('inner block left top point did not found.')
                return
            self.logger.info('inner block left top point found.')
            # 从数字块左上角出发，找到数字块下边界和右边界
            num_block_rb = [-1, -1]
            x, y = num_block_lt
            prev = -1
            cur = -1
            while num_block_rb[1] == -1 and y < img_bin.shape[0]:
                # if y >= img_bin.shape[0]:
                    # break
                y += 1
                prev = cur
                cur = img_bin[y][num_block_lt[0]]
                if self.__is_b2w(prev, cur):
                    num_block_rb[1] = y - 1
            if num_block_rb[1] == -1:
                self.logger.error('inner block bottom side did not found.')
                return
            self.logger.info('inner block bottom side found.')
            while num_block_rb[0] == -1 and x < img_bin.shape[1]:
                # if x >= img_bin.shape[1]:
                #     break
                x += 1
                prev = cur
                cur = img_bin[num_block_lt[1]][x]
                if self.__is_b2w(prev, cur):
                    num_block_rb[0] = x - 1
            if num_block_rb[0] == -1:
                self.logger.error('inner block right bottom did not found.')
                return
            self.logger.info('inner block right bottom found.')

            Detector.NUM_BLOCK_INFO = (num_block_lt[0], num_block_lt[1], num_block_rb[0], num_block_rb[1])
            self.logger.debug('NUM_BLOCK_INFO: %s' % ', '.join([str(t) for t in Detector.NUM_BLOCK_INFO]))
        except Exception:
            self.logger.exception('detect num block error. %s' % self.img_path)

    # 从数字块中识别数字字符串
    def detect(self):
        if Detector.NUM_BLOCK_INFO is None:
            raise RuntimeError('NUM_BLOCK_INFO is None')
        length = int(self.num_block.shape[1] / self.num_size)
        h = self.num_block.shape[0] / 9
        w = length / 3
        blocks = []
        for i in range(self.num_size):
            blocks.append(self.num_block[:, length * i:length * (i + 1)])
        for i, block in enumerate(blocks):
            # cv2.imwrite('block.bmp', block)
            # cv2.imshow('block', block)
            # cv2.waitKey()
            # return
            self.logger.debug('single num %d detecting' % (i+1, ))
            num = self.__detect(block, h, w)
            if num == '':
                self.nums = []
                self.logger.error('single num %d detect failed. %s' % (i+1, self.img_path))
                return
            self.logger.debug('single num %d detect success: %s' % (i+1, num))
            self.nums.append(num)

    @property
    def num_str(self):
        if not self.nums:
            return ''
        return ''.join(self.nums).zfill(self.zfill_width)

    # 识别单个数字
    def __detect(self, block, h, w):
        # self.logger.debug('w: %f, h: %f' % (w, h))
        bit_map = [0, 0, 0, 0, 0, 0, 0]
        for i in range(7):
            # cv2.imshow('block', block)
            # cv2.waitKey()
            # return
            try:
                bit_map[i] = self.__is_white(
                    block, int(math.ceil(self.CENTER_POS[i][0] * w)), int(math.ceil(self.CENTER_POS[i][1] * h)))
            except IndexError:
                self.logger.exception('Failed detect single num. Key point %d detect error.' % (i+1, ))
                return ''
            self.logger.debug('bit_map: %s' % bit_map)
        num = self.BITMAP.get(tuple(bit_map), '')
        self.logger.debug('bit_map: %s' % bit_map)
        return num

    # 判断当前像素点及其上、下、左、右的点是否有一个是白色
    @staticmethod
    def __is_white(img, x, y):
        try:
            pixel = img[y:y + 1, x:x + 1][0]
            pixel_left = img[y:y + 1, x - 1:x][0]
            pixel_right = img[y:y + 1, x + 1:x + 2][0]
            pixel_top = img[y - 1:y, x][0]
            pixel_bottom = img[y + 1:y + 2, x][0]
            area = [pixel_left, pixel_right, pixel_top, pixel_bottom, pixel]
        except IndexError:
            raise IndexError
        return 1 if [255] in area else 0

    # 判断当前像素点（黑）是否是从白到黑的突变点
    @staticmethod
    def __is_w2b(prev, cur):
        return prev == 255 and cur == 0

    # 判断当前像素点（白）是否是从黑到白的突变点
    @staticmethod
    def __is_b2w(prev, cur):
        return prev == 0 and cur == 255
