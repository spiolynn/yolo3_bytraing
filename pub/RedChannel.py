# -*- coding: utf-8 -*-
# @Time    : 2018/12/13 15:30
# @Author  : Shark
# @Site    : 
# @File    : RedChannel.py
# @Software: PyCharm Community Edition

'''
取红章
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def GetRedC(img):
    Bch, Gch, Rch = cv2.split(img)
    img_output = cv2.merge([Rch,Rch,Rch])
    return img_output


def HSV():

    image = cv2.imread(r'C:\004_project\012-yolo\AI_Training_yolo3\0_cut\201811056962367900000000000001-03M_0.jpg', cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([-40,70,70])
    upper_red = np.array([40,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_path_list(rootdir):
    '''
    :return: self.FilePathList
    '''
    FilePathList = []
    for fpathe, dirs, fs in os.walk(rootdir):
        for f in fs:
            FilePath = os.path.join(fpathe, f)
            if os.path.isfile(FilePath):
                FilePathList.append(FilePath)
    return FilePathList

def do():
    FileList = get_path_list('./0_cut')

    for f in FileList:
        #读入图像,三通道
        basename = os.path.basename(f)

        # timg.jpeg
        image=cv2.imread(f,cv2.IMREAD_COLOR)

        # 获得三个通道
        Bch,Gch,Rch=cv2.split(image)


        # 读入图像尺寸
        cols, rows, _ = image.shape
        # 红色通道的histgram
        # 变换程一维向量
        # pixelSequence = Rch.reshape([rows * cols, ])
        # # 统计直方图的组数
        # numberBins = 256
        # histogram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor='black',
        #                                   histtype='bar')  # facecolor设置为黑色
        #
        # # 设置坐标范围
        # y_maxValue = np.max(histogram)
        # # plt.figure()
        # plt.axis([0, 255, 0, y_maxValue])
        # # 设置坐标轴
        # plt.xlabel("gray Level", fontsize=20)
        # plt.ylabel('number of pixels', fontsize=20)
        # plt.title("Histgram of red channel", fontsize=25)
        # plt.xticks(range(0, 255, 10))
        # # 显示直方图
        # plt.savefig('./1_cut/' + basename.split('.')[0] + '_2' + '.jpg', dpi=260, bbox_inches="tight")
        # plt.close()

        # 红色通道阈值
        _,RedThresh = cv2.threshold(Rch,150,255,cv2.THRESH_BINARY)

        # 膨胀操作
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        erode = cv2.erode(RedThresh, element)

        # cv2.imwrite('./1_cut/0_' + basename,RedThresh)
        cv2.imwrite('./1_cut/' + basename.split('.')[0] + '_0' + '.jpg', Rch)
        # cv2.imwrite('./1_cut/' +basename.split('.')[0] + '_1' + '.jpg',erode)
