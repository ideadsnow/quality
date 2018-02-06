# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy
import csv
import cv2
import json

def main():
    video_name = sys.argv[1]
    align_result_path = os.path.abspath('%s_align_result.csv' % (video_name,))
    fr_result_path = os.path.abspath('%s_fr_result.csv' % (video_name,))
    target_row = int(sys.argv[2])
    # align_result_path = os.path.abspath(sys.argv[1])
    # fr_result_path = os.path.abspath(sys.argv[2])
    # target_row = int(sys.argv[3])

    with open(align_result_path, 'rb') as align_result_file, open(fr_result_path, 'rb') as fr_result_file:
        align_result_reader = csv.reader(align_result_file)
        fr_result_reader = csv.reader(fr_result_file)

        for i in range(target_row - 1):
            align_result_reader.next()
            fr_result_reader.next()

        align_result_row = align_result_reader.next()
        fr_result_row = fr_result_reader.next()

        merged_row = align_result_row[:2] + fr_result_row[:1]
        # print merged_row[0], merged_row[1], merged_row[2]
        os.system("start %s" % (merged_row[0],))
        os.system("start %s" % (merged_row[1],))

if __name__ == '__main__':
    main()