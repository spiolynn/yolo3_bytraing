# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 18:38
# @Author  : Shark
# @Site    : 
# @File    : Lkmeans.py
# @Software: PyCharm Community Edition

'''
行链接聚类算法
'''

# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 18:33
# @Author  : Shark
# @Site    :
# @File    : julei.py
# @Software: PyCharm Community Edition
 #-*- coding:utf-8 -*-


import math
import numpy as np
from config import YOLO3Config as C


def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def xx_intersection_over_union(boxA, boxB):
    boxA = boxA[0][3]
    boxB = boxB[0][3]
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    UxA = min(boxA[0], boxB[0])
    UxB = max(boxA[2], boxB[2])
    X_interArea = max(0, xB - xA + 1)
    if X_interArea == 0 :
        return 999999999
    X_Union = max(0,UxB-UxA)
    X_IoU = float(X_Union)/X_interArea
    return X_IoU

def xx_len_c(boxA,boxB):
    boxA = boxA[0][3]
    boxB = boxB[0][3]
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    len_a = boxA[2] - boxA[0]
    len_b = boxB[2] - boxB[0]

    if len_b/len_a > 1:
        return len_b/len_a
    else:
        return len_a/len_b

# 定义两个box之间的距离

def box_dist(boxA,boxB):
    x = xx_intersection_over_union(boxA,boxB) * xx_len_c(boxA,boxB)
    return x

#找到距离最小的下标
def find_Min(M):
    min = 1000
    x = 0; y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j];x = i; y = j
    return (x, y, min)

#算法模型：
def AGNES(dataset, dist, k):
    #初始化C和M
    C = [];M = []
    for i in dataset:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist(i, j))
        M.append(Mi)
    q = len(dataset)
    #合并更新
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)
        q -= 1
    return C


# C = AGNES(dataset, dist_avg, 3)


'''
y轴iou
'''
def yy_iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[1], boxB[1])
    xB = min(boxA[3], boxB[3])
    UxA = min(boxA[1], boxB[1])
    UxB = max(boxA[3], boxB[3])
    X_interArea = max(0, xB - xA + 1)
    if X_interArea == 0 :
        return 99
    X_Union = max(0,UxB-UxA)
    X_IoU = float(X_Union)/X_interArea
    return X_IoU

'''
box 之间的距离（相对）
'''
def xy_dist(boxA,boxB):
    pass
    boxA_right_down_x = boxA[2]
    boxA_right_down_y = boxA[3]

    boxB_left_down_x = boxB[0]
    boxB_left_down_y = boxB[3]

    base = boxA[3]-boxA[1]

    bian_1p = math.pow(boxB_left_down_x - boxA_right_down_x,2)
    bian_2p = math.pow(boxB_left_down_y - boxA_right_down_y,2)
    dist = math.sqrt(bian_1p+bian_2p)/base

    return dist

'''
计算两box左顶点距离
'''

def point_dist(point_x,point_y):

    bian_1p = math.pow(point_x[0] - point_y[0], 2)
    bian_2p = math.pow(point_x[1] - point_y[1], 2)
    dist = math.sqrt(bian_1p + bian_2p)
    return dist

'''
计算两个box 左顶点距离
'''
def ctpn_tu_dist(boxA,boxB):
    boxA_left_top_x = boxA[0]
    boxA_left_top_y = boxA[1]

    boxA_left_down_x = boxA[0]
    boxA_left_down_y = boxA[3]

    boxB_left_top_x = boxB[0]
    boxB_left_top_y = boxB[1]

    boxB_left_down_x = boxB[0]
    boxB_left_down_y = boxB[3]

    dist_1 = point_dist((boxA_left_top_x,boxA_left_top_y),(boxB_left_down_x,boxB_left_down_y))
    dist_2 = point_dist((boxA_left_down_x, boxA_left_down_y), (boxB_left_top_x, boxB_left_top_y))
    dist = min(dist_1,dist_2)/max(dist_1,dist_2)
    return  dist



def x_dist_rel(boxA,boxB):
    '''
    X轴相对位置
    '''

    boxA_W = boxA[2] - boxA[0]

    boxA_right_down_x = boxA[2]
    boxB_left_down_x = boxB[0]

    dist = float(abs(boxB_left_down_x-boxA_right_down_x)/boxA_W) # >1 排除

    return dist



# 框子的宽高比
def box_box(box):
    return (box[2]-box[0])/(box[3]-box[1])


def idokonw(title_box,ctpn_boxes,label_boxes):
    # STEP1: 去除宽高比小于1.6的框
    ctpn_boxes_new = []
    for j, ctpn_boxes_j in enumerate(ctpn_boxes):
        if box_box(ctpn_boxes_j) < C.Thresh_Bbox_WH:
            pass
        else:
            ctpn_boxes_new.append(ctpn_boxes_j)
    ctpn_boxes = ctpn_boxes_new

    # step2: 去除ctpn_boxes x轴距离超过一个身位的bbox(C.Thresh_Bbox_X_DIST_REL)
    dist_xx_map = 99999999 * np.zeros((len(title_box), len(ctpn_boxes))).astype(np.float32)
    dist_xx_map_TF = np.zeros((len(title_box), len(ctpn_boxes))).astype(np.int)

    for i,title_box_i in enumerate(title_box):
        for j,ctpn_boxes_j in enumerate(ctpn_boxes):
            dist_xx_map[i][j] = x_dist_rel(title_box_i,ctpn_boxes_j)
    dist_xx_map_TF[dist_xx_map > C.Thresh_Bbox_X_DIST_REL] = 1
    dist_xx_map_TFSUM = dist_xx_map_TF.sum(axis=0)
    dist_xx_map_TFSUM_TF = dist_xx_map_TFSUM != dist_xx_map_TF.shape[0]
    ctpn_boxes_new = []
    for j, ctpn_boxes_j in enumerate(ctpn_boxes):
        if dist_xx_map_TFSUM_TF[j]:
            ctpn_boxes_new.append(ctpn_boxes_j)
        else:
            pass
    ctpn_boxes = ctpn_boxes_new

    # step3 : 计算title_box 和 ctpn_box 的距离
    dist_xy_map = 99999999 * np.zeros((len(title_box), len(ctpn_boxes))).astype(np.float32)
    dist_yy_iou_map = 99999999 * np.zeros((len(title_box), len(ctpn_boxes))).astype(np.float32)
    for i,title_box_i in enumerate(title_box):
        for j,ctpn_boxes_j in enumerate(ctpn_boxes):
            # 两框距离
            dist_xy_dist = xy_dist(title_box_i,ctpn_boxes_j)
            # y轴的iou
            yy_iou_dist = yy_iou(title_box_i, ctpn_boxes_j)
            dist_xy_map[i][j] = dist_xy_dist
            dist_yy_iou_map[i][j] = yy_iou_dist
    # title_box 和 ctpn_box 的综合距离
    bbox_dist_map = dist_xy_map * dist_yy_iou_map

    # 为每一个title_box 找到一个距离最小的的ctpn box
    dist_xy_map = np.zeros((len(title_box), len(ctpn_boxes))).astype(np.float32)
    while True:
        dist_min = bbox_dist_map.min()
        if dist_min > 99:
            break
        dist_xy_map[bbox_dist_map == dist_min] = dist_min
        bbox_dist_map[bbox_dist_map == dist_min] = 9999999999
        titlebox_flag = dist_xy_map.sum(axis=1)
        a = titlebox_flag!=0
        a = a.tolist()
        if a == [True,True,True]:
            break


    # 最佳匹配CTPN_Box
    ctpn_boxes_best_choose = -1 * np.ones((len(title_box), 1))
    indexes = np.argwhere(dist_xy_map != 0 )
    if len(indexes.tolist()) != 0 :
        a = indexes[:, 0]
        b = indexes[:, 1]
        for i, title_box_i in enumerate(title_box):
            q = np.where(a==i)
            temp = b[q]
            if len(temp) != 0:
                ctpn_boxes_best_choose[i] = temp

    # 最佳匹配后的可能box

    # ctpn_box 之间的距离定义
    ctpn_box_dist_map = np.zeros((len(ctpn_boxes), len(ctpn_boxes))).astype(np.float32)
    for i, ctpn_box_i in enumerate(ctpn_boxes):
        for j, ctpn_boxes_j in enumerate(ctpn_boxes):

            # 计算两个box 左顶点相对距离
            aa = ctpn_tu_dist(ctpn_box_i, ctpn_boxes_j)
            # box的iou
            iou = bb_intersection_over_union(ctpn_box_i, ctpn_boxes_j)
            if iou > 0 and iou < 0.2:
                weight = 10
            else:
                weight = 1
            ctpn_box_dist_map[i,j] = aa/weight

    ctpn_boxes_good_choose = -1 * np.ones((len(title_box), 1))
    for i,best_box in enumerate(ctpn_boxes_best_choose):
        if best_box[0] == -1:
            continue
        else:
            ctpn_box_index = int(best_box[0])
            best_ctpn_best_dist = ctpn_box_dist_map[ctpn_box_index,:]
            anchar = np.argwhere(best_ctpn_best_dist<0.15)

            # 没有第二好的框框
            if anchar.tolist() == []:
                pass
            else:
                ctpn_boxes_good_choose[i] = anchar

    # ctpn_boxes_choose = []
    # for i, title_box_i in enumerate(title_box):
    #     q = np.where(dist_xy_map[i] != 0 )
    #     qshape = q[0].size
    #     if qshape !=0:
    #     # 表示有匹
    #         for i,index in enumerate(q[0]):
    #             ctpn_boxes_choose.append(i)
    #
    #     if len(ctpn_boxes_choose) < 2:
    #         ctpn_box_dist_min = ctpn_box_dist_map.min(axis=0)


    return (title_box,ctpn_boxes,ctpn_boxes_best_choose,ctpn_boxes_good_choose)


if __name__ == '__main__':

    title_box = [[0, 278, 360, 335], [2, 188, 369, 247], [25, 5, 349, 66]]
    ctpn_boxes = [[377, 109, 956, 159],[377, 164, 500, 200]]

    title_box = [[0, 278, 360, 335], [2, 188, 369, 247], [25, 5, 349, 66]]
    ctpn_boxes = [[377, 109, 956, 159], [377, 164, 500, 200],[377, 288, 508, 335], [377, 21, 883, 65], [377, 198, 1144, 238]]


    # ctpn_boxes = [[377, 109, 956, 159]]

    # title_box = np.array(title_box)
    # q = np.where(title_box[0] != 0)
    # print('sf')
    # pass
    idokonw(title_box,ctpn_boxes,[])





