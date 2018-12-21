# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 9:40
# @Author  : panzi
# @Site    : 
# @File    : config.py
# @Software: PyCharm Community Edition

'''
YOLO3 超参数信息
'''

class YOLO3Config:

    pass

    USE_GPU = False

    # 训练数据label文件
    annotation_path_xml = './model_data/9404-300.xml'
    annotation_path = './model_data/9404-300_train.txt'

    # 日志路径
    log_dir = 'logs/001/'

    # 分类标签
    classes_path = 'model_data/9404-300_classes.txt'

    # 锚
    anchors_path = 'model_data/9404-300_anchors.txt'

    # multiple of 32, hw
    input_shape = (416,416)

    # 预训练模型路径
    weights_path = 'model_data/yolo.h5'

    # 测试/训练数据比例
    val_split = 0.1

    # 一阶段batch_size
    fstep_batchsize = 32
    fepoch = 30

    # 二阶段batch_size
    sstep_batchsize = 8
    sepoch = 20

    gpu_id = '1'

    ########################## predict

    # 模型文件
    model_path = r'C:\004_project\012-yolo\AI_Training_yolo3\logs\001\trained_weights_final.h5'

    score = 0.3
    iou = 0.2

    # 聚类
    Kmeans_Class = 2

    #
    Cut_Pixel = 5

    # bbox排除
    Thresh_Bbox_WH = 1.6
    Thresh_Bbox_X_DIST_REL = 0.5