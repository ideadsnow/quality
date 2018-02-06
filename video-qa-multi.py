# -*- coding: utf-8 -*-
__author__ = 'LibX'

import os
import sys
import re
import time
import Queue
import threading
import subprocess
import hashlib
import csv
import logging
import shutil
import log

default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')

def config_logging(log_dir=default_log_dir):
    log_file_path = os.path.join(log_dir, 'video-qa-multi.log')
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(name)-20s %(levelname)-10s %(message)s',
                #datefmt='%a, %d %b %Y %H:%M:%S',
                filename=log_file_path,
                filemode='a')

def video_qa(video_file_path, image_basedir):
    '''
    评分
    '''
    logger = logging.getLogger('video_qa')
    cmd = ['python', r'C:\script_libx\video-qa.py', video_file_path, image_basedir]
    logger.debug('run video-qa cmd %s' % (' '.join(cmd), ))
    child = subprocess.Popen(cmd)
    child.wait()

class QualityWorkerThread(threading.Thread):
    def __init__(self, queue, image_basedir, logger=None):
        threading.Thread.__init__(self)
        self.queue = queue
        self.image_basedir = image_basedir
        self.logger = logger
    
    def debug(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.debug(*args, **kwargs)
    
    def info(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)
    
    def exception(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.exception(*args, **kwargs)
    
    def run(self):
        done = False
        try:
            while not done:
                try:
                    video_file_path = self.queue.get(True, 1)
                    
                    self.info('video-qa on %s' % (os.path.basename(video_file_path), ))
                    score = video_qa(video_file_path, self.image_basedir)
                    self.queue.task_done()
                except Queue.Empty:
                    self.debug('queue is empty')
                    done = True
            self.debug('worker finished')
        except:
            self.exception('worker error')

def video_qa_multi(video_basedir, image_basedir):
    logger = logging.getLogger('video_qa_multi')
    start_time = time.time()
    try:
        logger.info('video_qa_multi %s ' % (video_basedir, ))
        
        queue = Queue.Queue()
        for f in os.listdir(video_basedir):
            video_file_path = os.path.join(video_basedir, f)
            if os.path.isfile(video_file_path) and (video_file_path.endswith('.mp4')):
            
                queue.put(video_file_path)
        
        threads = []
        result = []
        for i in range(3):
            t = QualityWorkerThread(queue, image_basedir, logger=logger)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
    finally:
        end_time = time.time()
        logger.info('video_qa_multi %s spent %ds' % (video_basedir, (end_time-start_time) ))

if __name__=="__main__":
    video_basedir = sys.argv[1] # "H:\\finish\\"
    image_basedir = sys.argv[2] # 'G:\\'
    config_logging()
    video_qa_multi(video_basedir, image_basedir)
    
