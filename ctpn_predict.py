# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 14:38
# @Author  : Shark
# @Site    : 
# @File    : predict_zhizhao.py
# @Software: PyCharm Community Edition

'''
营业执照的文字定位
'''

'''
身份证定位
'''

import ctpn.id_card_ctpn as id_card_ctpn
import os
import matplotlib.pyplot as plt
from pub.TTime import exeTime
import cv2
import shutil
import numpy as np
from config_ctpn import CTPNConfig as C

VGG_PATH = C.VGG_PATH
CTPN_MODEL_PATH = C.CTPN_MODEL_PATH

# POS_PATH = './ctpn_result'
# if os.path.exists(POS_PATH):
#     shutil.rmtree(POS_PATH)
# os.mkdir(POS_PATH)
#
# CUT_PATH = './ctpn_result_cut'
# if os.path.exists(CUT_PATH):
#     shutil.rmtree(CUT_PATH)
# os.mkdir(CUT_PATH)


class id_card_word_position(object):
    def __init__(self):
        pass
        self.id_card_ctpn = id_card_ctpn.id_card_ctpn()
        self.id_card_ctpn.bulid_model(VGG_PATH)
        self.id_card_ctpn.load_model(CTPN_MODEL_PATH)

    def get_path_list(self, rootdir):
        self.FilePathList = []
        for fpathe, dirs, fs in os.walk(rootdir):
            for f in fs:
                FilePath = os.path.join(fpathe, f)
                if os.path.isfile(FilePath):
                    self.FilePathList.append(FilePath)
        return self.FilePathList

    @exeTime
    def predict(self, img):
        pass
        m_img, text, scale = self.id_card_ctpn.predict(img)
        return m_img, text,scale


    def predict_zhizhao(self,image,filepath=''):
        pass
        m_img, text, scale = self.predict(image)
        boxes = self.pick_box_zhizhao(text,m_img)

        BOXES = []
        for i, box in enumerate(boxes):
            XX1 = min(box[0][0], box[0][4])
            YY1 = min(box[0][1], box[0][3])
            XX2 = max(box[0][6], box[0][2])
            YY2 = max(box[0][7], box[0][5])
            BOXES.append([XX1,YY1,XX2,YY2])

            # print('ctpn')
            # print(m_img.shape)
            print([XX1,YY1,XX2,YY2])

        return (BOXES,m_img,scale)


    def pick_box_zhizhao(self, text, img):
        # 找到地址的box
        pass
        (H, W, C) = img.shape
        boxes = []
        max_w = 0
        for box in text:

            w = box[2] - box[0]
            if max_w < w:
                max_w = w
            h = box[5] - box[1]
            POINT_y = box[1]
            POINT_x = box[0]
            Flag = 0
            boxes.append([box, w, h, POINT_y, POINT_x, Flag])
        return boxes

if __name__ == '__main__':
    a_id_card_word_position = id_card_word_position()
    FilePathList = a_id_card_word_position.get_path_list(
        r'C:\004_project\012-yolo\AI_Training_yolo3\0_cut')

    for f_p in a_id_card_word_position.FilePathList:
        print('exec ' + f_p)
        img = cv2.imread(f_p)
        m_img, text = a_id_card_word_position.predict(img)

        # 对于text进行划分
        boxes = a_id_card_word_position.pick_box_zhizhao(text, m_img)

        m_img_copy = m_img.copy()
        m_img_copy_write = m_img.copy()
        basename = os.path.basename(f_p)
        savename = os.path.join(POS_PATH, basename)

        for i, box in enumerate(boxes):
            XX1 = min(box[0][0], box[0][4])
            YY1 = min(box[0][1], box[0][3])
            XX2 = max(box[0][6], box[0][2])
            YY2 = max(box[0][7], box[0][5])

            if abs(box[5]) == 2:
                cut_name = basename.split('.')[0] + '_' + str(i) + '.jpg'
                cut_name = os.path.join(CUT_PATH, cut_name)
                cut_img = m_img_copy_write[YY1:YY2, XX1:XX2]
                if box[5] == 2:
                    cv2.imwrite(cut_name, cut_img)
                else:
                    cut_img = np.rot90(cut_img)
                    cut_img = np.rot90(cut_img)
                    cv2.imwrite(cut_name, cut_img)

                cv2.rectangle(m_img_copy, (XX1, YY1),
                              (XX2, YY2), (0, 0, 255), 3)
            elif box[5] == 1:
                cv2.rectangle(m_img_copy, (XX1, YY1),
                              (XX2, YY2), (255, 0, 0), 2)
            else:
                cv2.rectangle(m_img_copy, (XX1, YY1),
                              (XX2, YY2), (0, 255, 0), 1)
        cv2.imwrite(savename, m_img_copy)

        # for i,box in enumerate(text):
        #     cv2.rectangle(m_img_copy, (min(box[0],box[4]), min(box[1],box[3])), (max(box[6],box[2]), max(box[7],box[5])), (255, 0, 0), 2)
        # cv2.imwrite(savename, m_img_copy)
