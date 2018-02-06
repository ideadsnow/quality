# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy
import subprocess
import csv
import logging
import cv2
import json
import math
import num_detector as detector
from configs import Settings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_LIB_PATH = os.path.join(os.path.dirname(BASE_DIR), 'video')
sys.path.append(VIDEO_LIB_PATH)

from utils import get_video_shape, VideoFrameGenerator, ROIFrameFilter, ResizeFrameFilter, find_bin
from blackdetection import AverageGrayThresholdDetector

__author__ = 'LibX'

logger = logging.getLogger('video_quality')
default_log_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'log')
SSIM_DEFAULT_KERNEL = cv2.getGaussianKernel(11, 1.5)
SSIM_DEFAULT_WINDOW = SSIM_DEFAULT_KERNEL * SSIM_DEFAULT_KERNEL.T


def config_logging(video_file_path, log_dir=default_log_dir):
    video_name = get_file_name(video_file_path)
    log_file_path = os.path.join(log_dir, '%s.log' % (video_name, ))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-20s %(levelname)-10s %(message)s',
                        filename=log_file_path,
                        filemode='a')


def get_file_name(video_file_path):
    video_file_name = os.path.basename(video_file_path)
    return os.path.splitext(video_file_name)[0]


def process_image_handler(error_csv_output_path, fr_csv_output_path, video_file_path, image_basedir, roi_setting, settings):
    frame_type = 'bmp'
    video_name = os.path.basename(video_file_path).split('.')[0]
    platform = video_name.split('_')[1]
    app_name = video_name.split('_')[0]
    # TODO 把参数获取方式整理好，现在好恶心
    pre_roi_frames_dict = None
    try:
        pre_roi_frames_dict = settings.get_specific_settings(
            platform, app_name, 'pre_roi'
        )
    except ValueError as e:
        logger.info('%s_%s no need pre roi' % (app_name, platform))

    resize_frames_dict = settings.get_specific_settings(
        platform, app_name, 'resize')
    video_shape = get_video_shape(video_file_path)
    # frame_rate = get_frame_rate(video_file_path)
    frame_rate = 60
    if not os.path.exists(image_basedir):
        os.makedirs(image_basedir)

    src_align_frames_dict = settings.get_specific_settings(
        platform, app_name, 'align')
    src_align_frames_path = src_align_frames_dict.get('align_src')
    fill_format = '%05d.'
    frames_file_pattern = os.path.join(
        image_basedir, video_name + fill_format + frame_type)
    src_frames_file_pattern = os.path.join(
        # src_align_frames_path, 'default' + fill_format + frame_type
        src_align_frames_path, '0' + fill_format + frame_type
    )

    x = roi_setting.get('x')
    y = roi_setting.get('y')
    width = roi_setting.get('width')
    height = roi_setting.get('height')

    if pre_roi_frames_dict:
        pre_x = pre_roi_frames_dict.get('x')
        pre_y = pre_roi_frames_dict.get('y')
        pre_width = pre_roi_frames_dict.get('width')
        pre_height = pre_roi_frames_dict.get('height')

    resize_width = resize_frames_dict.get('width')
    resize_height = resize_frames_dict.get('height')

    if not width:
        width = video_shape[0] - x
    if not height:
        height = video_shape[1] - y

    logger.info('use frame rate:{0} to extract video'.format(frame_rate))
    frame_generator = VideoFrameGenerator(
        video_file_path, video_shape, frame_rate)
    with frame_generator as frames:
        i = 1
        prev_frame = None
        # 数字区域坐标信息
        block_info = []
        with open(error_csv_output_path, 'wb') as error_csv, open(fr_csv_output_path, 'wb') as fr_csv:
            error_csv_writer = csv.writer(error_csv)
            fr_csv_writer = csv.writer(fr_csv)
            for frame in frames:
                should_write = True

                # 切黑边
                if pre_roi_frames_dict:
                    pre_roi_filter = ROIFrameFilter(
                        pre_x, pre_y, pre_width, pre_height)
                    frame = pre_roi_filter.filter(frame)
                # 放大
                if video_shape[0] != resize_width or video_shape[1] != resize_height:
                    resize_filter = ResizeFrameFilter(
                        resize_width, resize_height)
                    frame = resize_filter.filter(frame)
                # 切广告和水印
                roi_filter = ROIFrameFilter(x, y, width, height)
                roi_frame = roi_filter.filter(frame)

                if prev_frame is not None:
                    should_write = not numpy.array_equal(prev_frame, roi_frame)
                if should_write:
                    frame_file_name = frames_file_pattern % (i,)
                    black_detector = AverageGrayThresholdDetector(0)
                    if int(black_detector.get_avg_gray(roi_frame)) > 20:
                        cv2.imwrite(frames_file_pattern % (i,), roi_frame)
                        try:
                            # 识别失败会返回空字符串，int 强制转换会 raise exception
                            num = int(detector.Detector(
                                img=frame, logger=logger).num_str)

                            src_path = src_frames_file_pattern % (num,)
                            if not os.path.exists(src_path):
                                logger.error(
                                    'src pic %s is not exists.' % src_path)
                            target_path = frame_file_name
                            src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                            target = cv2.imread(
                                target_path, cv2.IMREAD_GRAYSCALE)
                            score = ssim(src, target, downsample=False)
                            row = [target_path, src_path,
                                   round(float(score), 5)]
                            fr_csv_writer.writerow(row)
                        except Exception:
                            logger.exception(
                                '%s detect number failed' % frame_file_name)
                            error_csv_writer.writerow([frame_file_name])
                prev_frame = roi_frame
                i += 1
    return [os.path.join(image_basedir, _) for _ in os.listdir(image_basedir) if os.path.isfile(os.path.join(image_basedir, _))]


def filter2valid(src, window):
    # https://cn.mathworks.com/help/matlab/ref/filter2.html#inputarg_shape
    ret = cv2.filter2D(src, -1, window, anchor=(1, 1),
                       delta=0, borderType=cv2.BORDER_CONSTANT)
    return ret[1:ret.shape[0] - window.shape[0] + 2, 1:ret.shape[1] - window.shape[1] + 2]


def ssim(img1, img2, K=(0.01, 0.03), window=SSIM_DEFAULT_WINDOW, L=255, downsample=True):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    assert(img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1])

    assert(len(K) == 2 and K[0] >= 0 and K[1] >= 0)

    M, N = img1.shape[0:2]
    H, W = window.shape[0:2]
    assert(H * W >= 4 and H <= M and W <= N)

    # automatic downsampling
    f = max(1, int(round(min(M, N) / 256.0)))
    # downsampling by f
    # use a simple low-pass filter
    if downsample and f > 1:
        lpf = numpy.ones((f, f))
        lpf = lpf / numpy.sum(lpf)

        # In opencv, filter2D use the center of kernel as the anchor,
        # according to http://docs.opencv.org/2.4.8/modules/imgproc/doc/filtering.html#void filter2D(InputArray src, OutputArray dst, int ddepth, InputArray kernel, Point anchor, double delta, int borderType)
        # but in matlab, imfilter use (2, 2) (matlab array starts with 1) as the anchor,
        # To ensure the results are the same with matlab's implementation, we
        # set filter2D's anchor to (1, 1) (python array starts with 0)
        img1 = cv2.filter2D(img1, -1, lpf, anchor=(1, 1),
                            borderType=cv2.BORDER_REFLECT)
        img2 = cv2.filter2D(img2, -1, lpf, anchor=(1, 1),
                            borderType=cv2.BORDER_REFLECT)

        img1 = img1[0::f, 0::f]
        img2 = img2[0::f, 0::f]

    C1, C2 = tuple((k * L) ** 2 for k in K)

    window = window / numpy.sum(window)

    mu1 = filter2valid(img1, window)
    mu2 = filter2valid(img2, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2

    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2valid(img1 * img1, window) - mu1_sq

    sigma2_sq = filter2valid(img2 * img2, window) - mu2_sq

    sigma12 = filter2valid(img1 * img2, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_scalar = cv2.mean(ssim_map)

    return ssim_scalar[0]


def process_ssim(align_csv_path, fr_output_csv_file):
    with open(align_csv_path, 'rb') as f:
        csv_data = [tuple(line[0:2]) for line in csv.reader(f)]

    with open(fr_output_csv_file, 'wb') as f:
        spam_writer = csv.writer(f)
        for target_image_path, src_image_path in csv_data:
            src_image = cv2.imread(src_image_path, cv2.IMREAD_GRAYSCALE)
            target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
            score = ssim(src_image, target_image, downsample=False)
            spam_writer.writerow(
                [target_image_path, src_image_path, round(float(score), 5)])


def align_frames(target_frames_path, align_csv_output_path, settings):
    executable_path = settings.get_specific_value('ALIGN_FRAMES_SCRIPT_PATH')
    base_frames_dir = os.path.basename(target_frames_path)
    platform = base_frames_dir.split('_')[1]
    app_name = base_frames_dir.split('_')[0]
    src_align_frames_dict = settings.get_specific_settings(
        platform, app_name, 'align')
    src_align_frames_path = src_align_frames_dict.get('align_src')
    logger.info('src align frames path:{0}'.format(src_align_frames_path))
    cmd = ['python', executable_path, src_align_frames_path, target_frames_path,
           '--output-file', align_csv_output_path, '--level', str(5)]
    child = subprocess.Popen(cmd)
    child.wait()


def get_frame_rate(video_path):
    fps = 60
    ffprobe_path = find_bin('ffprobe', environs=['FFMPEG_HOME'])
    cmd = [ffprobe_path,
           '-v', 'error',
           '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
           '-of', 'json',
           video_path]
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, _ = child.communicate()
    video_info = json.loads(stdout)
    stream_infos = video_info.get('streams', None)
    if stream_infos is None:
        return None

    for stream_info in stream_infos:
        r_frame_rate = stream_info.get('r_frame_rate', None)
        avg_frame_rate = stream_info.get('avg_frame_rate', None)
        if r_frame_rate is None or avg_frame_rate is None:
            continue
        # 判断是否是音频流，如果是舍弃
        if int(str(r_frame_rate).split('/')[1]) == 0:
            continue
        # 调用天花板函数处理数据

        h_r_frame_rate = math.ceil(parse_rate(str(r_frame_rate)))
        h_avg_frame_rate = math.ceil(parse_rate(str(avg_frame_rate)))

        # 判断是否是固定帧率，如果不是返回帧率120
        if h_r_frame_rate == h_avg_frame_rate:
            return int(h_r_frame_rate)
        return fps
    return None


def parse_rate(arg_str):
    top = float(arg_str.split('/')[0])
    down = float(arg_str.split('/')[1])
    return top / down


def video_quality(video_file_path, frame_basedir, roi_setting, csv_basedir, settings):

    start_time = time.time()
    try:
        video_name = get_file_name(video_file_path)
        logger.info('filename: %s ' % (video_name))

        csv_output_path = os.path.join(
            os.path.dirname(video_file_path), 'results')

        if not os.path.isdir(csv_output_path):
            os.mkdir(csv_output_path)

        error_csv_file_name = os.path.basename(
            frame_basedir) + '_error.csv'
        error_csv_output_path = os.path.join(
            csv_output_path, error_csv_file_name)
        reuslt_file_name = os.path.basename(frame_basedir) + '_fr_result.csv'
        fr_csv_output_path = os.path.join(csv_output_path, reuslt_file_name)

        # 解帧、识别数字、全参
        frame_paths = process_image_handler(
            error_csv_output_path, fr_csv_output_path, video_file_path, frame_basedir, roi_setting, settings)
        logger.info('process_image_handler finished!')

        # 对帧
        # align_frames(frame_basedir, align_csv_output_path, settings)
        # logger.info('align_frames finished!')

        # 全参
        # if not os.path.exists(align_csv_output_path):
        # logger.error('align_csv_output_path not exist! {0}'.format(
        # align_csv_output_path))
        # return

        # process_ssim(align_csv_output_path, fr_csv_output_path)
        logger.info('process_ssim finished!')

    except Exception, e:
        logger.exception(e)
    finally:
        end_time = time.time()
        logger.info('%s spent %ds' % (video_name, (end_time - start_time)))


def main():
    video_file_path = sys.argv[1]
    frame_basedir = sys.argv[2]
    csv_basedir = sys.argv[4]
    roi_setting = sys.argv[3]
    settings = Settings()
    config_logging(video_file_path)
    video_quality(video_file_path, frame_basedir,
                  eval(roi_setting), csv_basedir, settings)


if __name__ == "__main__":
    main()
