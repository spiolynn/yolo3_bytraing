# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 10:22
# @Author  : panzi
# @Site    : xml解析
# @File    : gen_yolo_labeltxt.py
# @Software: PyCharm Community Edition

'''
处理cvat 工具生成的label xml 转化为yolo格式
'''

import xml.etree.ElementTree as ET
from os import getcwd
import os
from config import YOLO3Config as C
import numpy as np

def _main(filename,label_dicts,output_file):
    wd = getcwd()
    file = filename
    filename = os.path.join(wd,file)

    print(filename)

    infile = open(filename)
    tree = ET.parse(infile)
    root = tree.getroot()

    label_dicts=label_dicts

    f = open(output_file,'w',encoding='utf-8')

    for image_i in root.iter('image'):
        textline = ""
        image_file_name = image_i.attrib['name']
        textline = textline + image_file_name
        for bbox in image_i.iter('box'):
            label = bbox.attrib['label']
            label = label_dicts[label]
            x_min = int(float(bbox.attrib['xtl']))
            y_min = int(float(bbox.attrib['ytl']))
            x_max = int(float(bbox.attrib['xbr']))
            y_max = int(float(bbox.attrib['ybr']))
            textline = textline + " " + str(x_min) + "," \
                       + str(y_min) + ","+str(x_max) + "," + str(y_max)+","+str(label)
        f.writelines(textline)
        f.writelines('\n')
        print(textline)
    f.close()

'''
对标记数据进行校验
'''
def label_check():

    label_lists = np.zeros((6, 1))

    output_file_problem = './model_data/20181219/problem1.txt'
    output_file_ok = './model_data/20181219/ok4.txt'

    f_problem = open(output_file_problem, 'w', encoding='utf-8')
    f_ok = open(output_file_ok, 'w', encoding='utf-8')

    file = C.annotation_path
    if not os.path.exists(file):
        print(file + " not exsits ")
        return 1
    else:
        with open(file,mode='r',encoding='utf-8') as f:
            for i,line in enumerate(f):
                label_lists = np.zeros((6, 1))
                line_list = line.split(' ')
                label_count = len(line_list)
                if label_count != 6:
                    f_problem.writelines(line)
                    f_problem.writelines('label not 6 ')
                else:
                    for list_i in range(1,label_count):
                        # 遍历标记
                        label = int(line_list[list_i].split(',')[4])
                        if label >= 0 and label <=4 :
                            # BusinessLicense
                            label_lists[0] = label_lists[0] + 1
                        elif label>=5 and label <=6:
                            # Company
                            label_lists[1] = label_lists[1] + 1
                        elif label>=7 and label<=13:
                            label_lists[2] = label_lists[2] + 1
                        elif label>=14 and label<=20:
                            label_lists[3] = label_lists[3] + 1
                        elif label>=21 and label<=23:
                            label_lists[4] = label_lists[4] + 1
                        elif label==24:
                            label_lists[5] = label_lists[4] + 1
                    if label_lists[5] != 0:
                        # 有error
                        f_problem.writelines(line)
                        f_problem.writelines('error_image')
                    elif label_lists.sum() != 5:
                        # 少标了
                        a = label_lists.sum()
                        f_problem.writelines(line)
                        f_problem.writelines('less 5')
                    elif (label_lists>2).any():
                        # 相同类型标记重复
                        a = label_lists>2
                        f_problem.writelines(line)
                        f_problem.writelines('duplica')
                    else:
                        f_ok.writelines(line)
                        # f_ok.writelines('\n')

                print(line_list)

    f_ok.close()
    f_problem.close()


if __name__ == "__main__":

    label_check()


    # label_file = C.classes_path
    #
    # label_dicts = {}
    # with open(label_file,encoding='utf-8',mode='r') as f:
    #     for i,s in enumerate(f):
    #         s = s.replace('\n','')
    #         label_dicts[s] = i
    #
    # print(label_file)
    # # label_dicts={
    # #     'person': 0 ,
    # #     'car' : 1 ,
    # #     'other' : 2
    # # }
    #
    # xmlfile = C.annotation_path_xml
    # output_file = C.annotation_path
    #
    # _main(xmlfile,label_dicts,output_file)

