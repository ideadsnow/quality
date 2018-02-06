# -*- coding: utf-8 -*-

'''
    重要！！！
    部署新机器的时候一定要注意根据情况修改以下目录配置
'''

import os
BASE_DEV_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

BASE_VIDEOS_DIR = 'fr_videos'
BASE_SRC_FRAMES_DIR = 'source'

WEB = {
    'default': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 12, 'y': 171, 'width': 1896, 'height': 807},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'web_default')}
    },
}

ANDROID = {
    'default': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 359, 'y': 161, 'width': 1547, 'height': 846},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'android_default')}
    },
    'csassist': {
        'pre_roi': {'x': 5, 'y': 0, 'width': 853, 'height': 480},
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 0, 'y': 163, 'width': 1918, 'height': 820},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_android')}
    },
    'dyassist': {
        'pre_roi': {'x': 0, 'y': 1, 'width': 848, 'height': 477},
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 0, 'y': 163, 'width': 1918, 'height': 820},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_android')}
    },
    'hyassist': {
        'resize': {'width': 1920, 'height': 1066},
        'roi': {'x': 0, 'y': 156, 'width': 1918, 'height': 820},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_android')}
    },
    'pdassist': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 0, 'y': 163, 'width': 1918, 'height': 820},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_android')}
    },
    'yyassist': {
        'resize': {'width': 1920, 'height': 1066},
        'roi': {'x': 0, 'y': 156, 'width': 1918, 'height': 820},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_android')}
    },
}

IOS = {
    'default': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 359, 'y': 161, 'width': 1547, 'height': 846},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'android_default')} # android 和 ios 切图参数一致，所以用同样的原图
    },
    'csassist': {
        'pre_roi': {'x': 1, 'y': 0, 'width': 851, 'height': 480},
        'resize': {'width': 1919, 'height': 1080},
        'roi': {'x': 0, 'y': 160, 'width': 1919, 'height': 752},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_ios')}
    },
    # 斗鱼 2017.12 月不测，以后要测需要确认参数
    'dyassist': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 1, 'y': 160, 'width': 1919, 'height': 752},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_ios')}
    },
    'hyassist': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 1, 'y': 160, 'width': 1919, 'height': 752},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_ios')}
    },
    'yyassist': {
        'resize': {'width': 1920, 'height': 1080},
        'roi': {'x': 1, 'y': 160, 'width': 1919, 'height': 752},
        'align': {'align_src': os.path.join(BASE_SRC_FRAMES_DIR, 'assist_ios')}
    },
}

VIDEO_SCRIPTS_DIR = os.path.join(BASE_DEV_DIR, 'video')
VIDEO_EXT = ['.mp4', '.avi', '.mov', '.flv']
AVG_GRAY_SCRIPT_PATH = os.path.join(VIDEO_SCRIPTS_DIR, 'avg_gray.py')
ALIGN_FRAMES_SCRIPT_PATH = os.path.join(VIDEO_SCRIPTS_DIR, 'align_frames.py')
SSIM_SCRIPT_PATH = os.path.join(VIDEO_SCRIPTS_DIR, 'ssim.py')
AVG_GRAY_THRESHOLD = 2
FR_FINISHED_VIDEOS = os.path.join(BASE_VIDEOS_DIR, 'fr_videos_finished.txt')
