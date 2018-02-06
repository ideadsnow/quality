# -*- coding:utf-8 -*-
import os
import importlib


class Settings(object):
    def __init__(self):
        module_name = os.environ.get(
            "FR_CONFIGS_MODULE", "configs.fr_settings")
        self.module = importlib.import_module(module_name)
        self.default_settings = {}
        self.specific_settings = {}
        self.__read_settings_to_dict()

    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            new = super(Settings, cls)
            cls._instance = new.__new__(cls, *args, **kw)
        return cls._instance

    def get_finished_videos(self):
        finished_videos_list = []
        file_list_path = self.get_specific_value('FR_FINISHED_VIDEOS')
        if not os.path.exists(file_list_path):
            with open(file_list_path, 'w') as f:
                pass
        with open(file_list_path, 'rb') as f:
            for line in f.readlines():
                finished_videos_list.append(line.strip())
        return finished_videos_list

    def update_finished_videos_list(self, finished_video_files):
        file_list_path = self.get_specific_value('FR_FINISHED_VIDEOS')
        with open(file_list_path, 'wb') as f:
            for finished_video in finished_video_files:
                f.write(finished_video + '\r\n')

    def get_specific_value(self, attr_name):
        return getattr(self.module, attr_name)

    def get_specific_settings(self, platform, app_name, settings_type):
        '''
        根据平台、应用名和配置类型，从配置文件中返回具体的配置信息，如下所示：
        roi:{'y': 189, 'x': 311, 'height': 726, 'width': 1220}
        align:{'align_src': 'D:\\quality_final_source\\category_A'}
        :param platform:
        :param app_name:
        :param settings_type:
        :return: dict
        '''
        platform_value_dict = self.specific_settings.get(platform.upper())
        if not platform_value_dict:
            result_value_dict = self.__get_default_settings(
                platform, settings_type)
            if not result_value_dict:
                raise ValueError(
                    'platform {0} config does not exist'.format(platform.upper()))
            return result_value_dict

        app_value_dict = platform_value_dict.get(app_name)
        result_value_dict = app_value_dict.get(
            settings_type) if app_value_dict else None

        if not result_value_dict:
            result_value_dict = self.__get_default_settings(
                platform, settings_type)

        if not result_value_dict:
            raise ValueError(
                'platform {0} app {1} config does not exist'.format(platform, app_name))

        return result_value_dict

    def __get_default_settings(self, platform, settings_type):
        platform_value_dict = self.default_settings.get(platform.upper())
        if not platform_value_dict:
            return None
        return platform_value_dict.get(settings_type)

    def __read_settings_to_dict(self):
        for settings in dir(self.module):
            if not settings.isupper():
                continue
            tmp_settings = {}
            settings_dict = getattr(self.module, settings)
            if not isinstance(settings_dict, dict):
                continue
            if 'default' in settings_dict:
                self.default_settings[settings] = settings_dict.get('default')
                settings_dict.pop('default')
            for key in settings_dict.keys():
                tmp_settings[key] = settings_dict.get(key)
            self.specific_settings[settings] = tmp_settings
