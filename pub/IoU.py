# -*- coding: utf-8 -*-
# @Time    : 2018/12/13 11:35
# @Author  : Shark
# @Site    : 
# @File    : IoU.py
# @Software: PyCharm Community Edition


'''
IoU = Area of OverLap / Area of Union
'''

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
'''
X轴 IoU
'''
def xx_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    UxA = min(boxA[0], boxB[0])
    UxB = max(boxA[2], boxB[2])
    X_interArea = max(0, xB - xA + 1)
    X_Union = max(0,UxB-UxA)
    X_IoU = X_interArea/float(X_Union)
    return X_IoU

def xx_len_c(boxA,boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    len_a = boxA[2] - boxA[0]
    len_b = boxB[2] - boxB[0]

    if len_b/len_a > 1:
        return len_a/len_b
    else:
        return len_b/len_a
