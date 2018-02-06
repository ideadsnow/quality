# -*- coding:utf-8 -*-
from configs import Settings
import os
import time
import sys
import Queue
import threading
import logging
import subprocess
import argparse
import math

default_log_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'log')


def config_logging(log_dir=default_log_dir):
    log_file_path = os.path.join(log_dir, 'video-qa-multi.log')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-20s %(levelname)-10s %(message)s',
                        filename=log_file_path,
                        filemode='a')


class FRProcessor(object):
    def __init__(self, fr_root_video_path, fr_root_frames_path):
        self.logger = logging.getLogger('FRProcessor')
        self.fr_root_video_path = fr_root_video_path
        self.fr_root_frames_path = fr_root_frames_path
        self.fr_videos_ready = []
        self.pid_file_name = os.path.basename(
            os.path.splitext(__file__)[0]) + '.pid'
        self.pid_file_path = os.path.join(
            self.fr_root_video_path, self.pid_file_name)
        self.settings = Settings()
        self.fr_finished_videos = self.settings.get_finished_videos()
        self.queue = Queue.Queue()

    def process(self):
        try:
            self.__setup()
            self.__get_ready_files()
            self.__video_qa_multi()
            self.__update_finished_videos()
        finally:
            # 最后执行完成后删除pid file
            self.__teardown()

    def __setup(self):
        # singleton
        if os.path.exists(self.pid_file_path):
            self.logger.error(
                'pid file is exist! {0}'.format(self.pid_file_path))
            sys.exit(1)

        with open(self.pid_file_path, 'w') as f:
            f.write(str(os.getpid()))

    def __teardown(self):
        if os.path.exists(self.pid_file_path):
            self.logger.info(
                'deleting the pid file:{0}'.format(self.pid_file_path))
            os.remove(self.pid_file_path)

    def __get_ready_files(self):
        video_ext_list = self.settings.get_specific_value('VIDEO_EXT')
        for parent, dirs, files in os.walk(self.fr_root_video_path):
            for _file in files:
                file_path = os.path.join(parent, _file)
                if os.path.splitext(file_path)[1] in video_ext_list:
                    try:
                        with open(file_path, 'rb') as f:
                            pass

                        self.logger.info(
                            "{0} file is ready!".format(file_path))
                        if file_path not in self.fr_finished_videos:
                            self.fr_videos_ready.append(file_path)
                    except IOError:
                        pass

    def __update_finished_videos(self):
        video_ext_list = self.settings.get_specific_value('VIDEO_EXT')
        result_list = []
        video_list = []
        finished_video_list = []
        for parent, dirs, files in os.walk(self.fr_root_video_path):
            for _file in files:
                if os.path.splitext(_file)[1] in video_ext_list:
                    video_list.append(os.path.join(parent, _file))

                if _file.find('fr_result') != -1:
                    result_list.append(_file)

        for _result in result_list:
            for _video in video_list:
                if _video.find(_result.replace('_fr_result.csv', '')) != -1:
                    finished_video_list.append(_video)
        self.settings.update_finished_videos_list(finished_video_list)

    def __video_qa_multi(self):
        if not self.fr_videos_ready:
            self.logger.info('no videos need to extract frames!')
            sys.exit(1)

        while self.fr_videos_ready:
            video_path = self.fr_videos_ready.pop(0)
            try:
                with open(video_path, 'rb') as f:
                    pass
            except IOError:
                continue
            self.queue.put(video_path)

        threads = []
        start_time = time.time()
        try:
            for i in range(3):
                t = QualityWorkerThread(self.queue, self.fr_root_video_path,
                                        self.settings, self.fr_root_frames_path, logger=self.logger)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
        finally:
            end_time = time.time()
            self.logger.info('video_qa_multi %s spent %ds' %
                             (self.fr_root_video_path, (end_time - start_time)))


class QualityWorkerThread(threading.Thread):
    def __init__(self, queue, fr_root_video_path, fr_setting, fr_root_frames_path, logger=None):
        threading.Thread.__init__(self)
        self.fr_root_video_path = fr_root_video_path
        self.fr_root_frames_path = fr_root_frames_path
        self.queue = queue
        self.logger = logger
        self.fr_setting = fr_setting

    def video_qa(self, video_file_path, image_basedir, roi_setting, result_csv_path):
        self.logger.info('video-qa on %s' %
                         (os.path.basename(video_file_path)))
        executable_path = os.path.join(
            os.path.dirname(__file__), 'video-qa.py')
        cmd = ['python', executable_path, video_file_path,
               image_basedir, str(roi_setting), result_csv_path]
        child = subprocess.Popen(cmd)
        child.wait()

    def run(self):
        done = False
        try:
            while not done:
                video_file_path = None
                try:
                    video_file_path = self.queue.get(True, 1)
                    if video_file_path is None:
                        continue

                    tmp_frames_path = str(video_file_path).replace(
                        self.fr_root_video_path, self.fr_root_frames_path)
                    current_frames_dir = os.path.splitext(tmp_frames_path)[0]
                    result_csv_path = os.path.join(
                        os.path.dirname(video_file_path), 'results')
                    file_name = os.path.basename(video_file_path)
                    app_name = file_name.split('_')[0]
                    platform = file_name.split('_')[1]
                    roi_setting = self.fr_setting.get_specific_settings(
                        platform, app_name, 'roi')

                    self.video_qa(video_file_path, current_frames_dir,
                                  roi_setting, result_csv_path)
                    self.queue.task_done()
                except Queue.Empty:
                    self.logger.debug('queue is empty')
                    done = True
                except Exception, e:
                    self.logger.exception(
                        'video_file_path: {0}'.format(video_file_path))
            self.logger.debug('worker finished')
        except Exception, e1:
            self.logger.exception('worker error', e1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fr_root_video_path', help='video source path')
    parser.add_argument('fr_root_frames_path', help='frames path')
    args = parser.parse_args()
    fr_root_video_path = args.fr_root_video_path
    fr_root_frames_path = args.fr_root_frames_path
    fr_processor = FRProcessor(fr_root_video_path, fr_root_frames_path)
    fr_processor.process()


if __name__ == '__main__':
    config_logging()
    main()
