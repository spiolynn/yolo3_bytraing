# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 10:50
# @Author  : panzi
# @Site    : yolo模型预测
# @File    : yolo3_predict.py
# @Software: PyCharm Community Edition

import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from keras.utils import plot_model
import cv2
from PIL import Image
from config import YOLO3Config as C
import tensorflow as tf
from pub.Lkmeans import AGNES,box_dist,idokonw,idokonw_v2
from pub.RedChannel import GetRedC
import ctpn_predict
import pprint
pp = pprint.PrettyPrinter(indent=4)


class YOLO3:

    def __init__(self):

        if C.USE_GPU:
            # 如果使用GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_id  # 使用 GPU 0和1
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)

        self.model_path = C.model_path
        self.anchors_path = C.anchors_path
        self.classes_path = C.classes_path
        self.score = C.score
        self.iou = C.iou
        self.model_image_size = C.input_shape

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.colors = self.__get_colors(self.class_names)


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    @staticmethod
    def __get_colors(names):
        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / len(names), 1., 1.)
                      for x in range(len(names))]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 转换~
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)  # anchors的数量
        num_classes = len(self.class_names)  # 类别数


        self.yolo_model = yolo_body(Input(shape=(416, 416, 3)), 3, num_classes)
        self.yolo_model.load_weights(model_path)  # 加载模型参数

        print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))

        # 根据检测参数，过滤框
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, self.anchors, len(self.class_names),
            self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def predict(self,image,image_path=''):
        '''
        真正的预测函数
        :param image:
        :return:
        '''
        start = timer()  # 起始时间
        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        print('detector size {}'.format(image_data.shape))
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        result = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]  # 类别
            box = out_boxes[i]  # 框
            score = out_scores[i]  # 执行度
            label = '{} {:.2f}'.format(predicted_class, score)  # 标签
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))  # 边框
            result.append([c,predicted_class,score,[left, top, right, bottom]])
        return (image_path,result)

    def detect_image(self, image):
        start = timer()  # 起始时间

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        print('detector size {}'.format(image_data.shape))
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 检测出的框

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(1e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 777  # 厚度
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]  # 类别
            box = out_boxes[i]  # 框
            score = out_scores[i]  # 执行度

            label = '{} {:.2f}'.format(predicted_class, score)  # 标签
            draw = ImageDraw.Draw(image)  # 画图
            label_size = draw.textsize(label, font)  # 标签文字

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))  # 边框

            if top - label_size[1] >= 0:  # 标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):  # 画框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(  # 文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
            del draw

        end = timer()
        print(end - start)  # 检测执行时间
        return image

    def close_session(self):
        self.sess.close()



'''
后处理将yolo定位关键位置图片切成目标区
单链接聚类
'''
def after_process(image_path, result):

    '''
    reuslt:
    0 = {list} <class 'list'>: [13, 'CreditCode_1', 0.9705741, [792, 748, 979, 814]]
    1 = {list} <class 'list'>: [8, 'Person_1', 0.99604374, [217, 1135, 577, 1192]]
    2 = {list} <class 'list'>: [3, 'Address_1', 0.99387616, [219, 1045, 586, 1104]]
    3 = {list} <class 'list'>: [2, 'Company_1', 0.9963999, [242, 862, 566, 923]]
    4 = {list} <class 'list'>: [0, 'BusinessLicense_1', 0.9928785, [350, 503, 1293, 700]]

    output:
    [result_1,CUT(4坐标)]
    '''

    basename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    (h,w,c) = img.shape

    need_box = []
    # 排除标题
    for res in result:
        if res[1].split('_')[0] != 'BusinessLicense': # and res[1].split('_')[0] != 'CreditCode' :
            need_box.append(res)

    # 当除标题外没有找到信息，则返回
    if need_box==[]:
        after_process_result = []
        return (image_path, after_process_result)

    result = AGNES(need_box,box_dist,C.Kmeans_Class)

    print('单链接聚类结果:\n')
    # pp.pprint(result)

    after_process_result = []
    for class_i in range(len(result)):
        need_box = []
        Cut_Long = False
        for box_i in result[class_i]:
            need_box.append(box_i[3])
            if box_i[1].split('_')[0] == 'CreditCode':
                Cut_Long = True
        need_box_np = np.array(need_box)
        print(need_box_np,need_box_np.shape)
        min_left = need_box_np.min(axis=0)[0]
        min_top = need_box_np.min(axis=0)[1]
        max_right = need_box_np.max(axis=0)[2]
        min_right = need_box_np.min(axis=0)[2]
        max_down = need_box_np.max(axis=0)[3]
        if Cut_Long:
            # 保留title
            cut_box = (max(min_top-C.Cut_Pixel,0),min(max_down+C.Cut_Pixel,h),min_left,w) # (y_min,y_max,x_min,x_max)
            cut_img = img[cut_box[0]:cut_box[1], cut_box[2]:cut_box[3]]
            for box_update_i in result[class_i]:
                # 更新yolo box的相对位置
                box_update_i[3][0] = box_update_i[3][0] - cut_box[2]
                box_update_i[3][2] = box_update_i[3][2] - cut_box[2]
                box_update_i[3][1] = box_update_i[3][1] - cut_box[0]
                box_update_i[3][3] = box_update_i[3][3] - cut_box[0]
            after_process_result.append([result[class_i],cut_img])
        else:
            cut_box = (max(min_top-C.Cut_Pixel,0), min(max_down+C.Cut_Pixel,h), min_left, w)
            cut_img = img[cut_box[0]:cut_box[1], cut_box[2]:cut_box[3]]
            for box_update_i in result[class_i]:
                # 更新yolo box的相对位置
                box_update_i[3][0] = box_update_i[3][0] - cut_box[2]
                box_update_i[3][2] = box_update_i[3][2] - cut_box[2]
                box_update_i[3][1] = box_update_i[3][1] - cut_box[0]
                box_update_i[3][3] = box_update_i[3][3] - cut_box[0]
            after_process_result.append([result[class_i], cut_img])

        # cv2.imwrite(str(class_i)+basename, cut_img)

    return (image_path,after_process_result)


def after_process_ctpn(ctpn, image_path, result):
    '''
    ctpn处理
    :param after_process_result:
    :return:
    '''

    i = 3
    basename = os.path.basename(image_path)

    for pic_i in result:
        # 分别处理一张图
        image = pic_i[1]
        m_img_copy = image.copy()
        (h, w, c) = image.shape
        # yolo定位信息 `[[18, 'CreditCode_1', 0.9990651, [0, 5, 300, 49]],...]`
        info = pic_i[0]
        title_box = []
        title_classname = []
        title_score = []
        for info_i in info:
            title_box.append(info_i[3])
            title_classname.append(info_i[1])
            title_score.append(info_i[2])
            # cv2.rectangle(m_img_copy, (info_i[3][0], info_i[3][1]),
            #               (info_i[3][2], info_i[3][3]), (0, 255, 0), 2)

        # box 四坐标
        title_box_np = np.array(title_box)
        # 找到坐标中[Ltop_x,Ltop_y,Rdown_x,Rdown_y] Rdown_x最小值
        min_right = title_box_np.min(axis=0)[2]

        cut_y_min = 0
        cut_y_max = h
        cut_x_min = min_right + 1
        cut_x_max = w
        image_cut = image[cut_y_min:cut_y_max, cut_x_min:cut_x_max]
        # 调用函数
        boxes, m_img, scale = ctpn.predict_zhizhao(GetRedC(image_cut))
        print('ctpn 结果:\n')
        pp.pprint(boxes)

        ctpn_boxes = []
        for box in boxes:
            print(box)
            x1 = int(box[0] / scale)
            y1 = int(box[1] / scale)
            x2 = int(box[2] / scale)
            y2 = int(box[3] / scale)
            x1 = x1 + cut_x_min
            x2 = x2 + cut_x_min
            ctpn_boxes.append([x1,y1,x2,y2])
            # cv2.rectangle(m_img_copy, (x1, y1),
            #               (x2, y2), (0, 0, 255), 3)



        print('ctpn after: title_box and ctpn_boxes\n')
        print(title_box,ctpn_boxes)
        m_img_copy_old = image.copy()
        for box_i,box in enumerate(title_box):
            cv2.rectangle(m_img_copy_old, (box[0], box[1]),
                          (box[2], box[3]), (0, 0, 255), 3)
        for box_i,box in enumerate(ctpn_boxes):
            cv2.rectangle(m_img_copy_old, (box[0], box[1]),
                          (box[2], box[3]), (255, 0, 0), 3)
        cv2.imwrite(str(i) + '1_ctpn_' + basename, m_img_copy_old)


        title_box, ctpn_boxes = idokonw_v2(title_box, ctpn_boxes, title_classname)
        m_img_copy_old1 = image.copy()
        for box_i,box in enumerate(title_box):
            cv2.rectangle(m_img_copy_old1, (box[0], box[1]),
                          (box[2], box[3]), (0, 0, 255), 3)
        for box_i,box in enumerate(ctpn_boxes):
            cv2.rectangle(m_img_copy_old1, (box[0], box[1]),
                          (box[2], box[3]), (255, 0, 0), 3)

            (h1,w1,c1) = image.shape
            cv2.imwrite('./cut/' + str(i) + '2_ctpn_' + str(box_i) + basename,image[max(box[1],0):min(box[3],h1),max(box[0],0):min(box[2],w1)])
        cv2.imwrite(str(i) + '2_ctpn_' + basename, m_img_copy_old1)


        ## 算法: 近邻box
        # title_box, ctpn_boxes,ctpn_boxes_best_choose,ctpn_boxes_good_choose = idokonw(title_box,ctpn_boxes,title_classname)
        # colors = [(0, 0, 255),(0, 255, 255),(0, 255, 0),(255, 0, 0),(255, 255, 0),(255, 255, 255)]
        #
        #
        #
        # for box_i,box in enumerate(title_box):
        #     cv2.rectangle(m_img_copy, (box[0], box[1]),
        #                   (box[2], box[3]), colors[box_i], 3)
        #     if ctpn_boxes_best_choose[box_i][0] != -1:
        #         box = ctpn_boxes[int(ctpn_boxes_best_choose[box_i][0])]
        #         cv2.rectangle(m_img_copy, (box[0], box[1]),
        #                       (box[2], box[3]), colors[box_i], 3)
        #     if ctpn_boxes_good_choose[box_i][0] != -1:
        #         box = ctpn_boxes[int(ctpn_boxes_good_choose[box_i][0])]
        #         cv2.rectangle(m_img_copy, (box[0], box[1]),
        #                       (box[2], box[3]), colors[box_i], 1)
        # # for box in ctpn_boxes:
        # #     cv2.rectangle(m_img_copy, (box[0], box[1]),
        # #                   (box[2], box[3]), (0, 255, 0), 3)
        #
        # pass
        # cv2.imwrite(str(i) +'_ctpn_' + basename, m_img_copy)


        i = i + 1




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

###################################################### test


def detect_img_for_batch_predict():

    yolo = YOLO3()
    ctpn = ctpn_predict.id_card_word_position()

    FileList = get_path_list(C.FileList)
    for f in FileList:
        print('handle: ' + f)
        image = Image.open(f)
        (image_path, result) = yolo.predict(image, image_path=f)
        print('yolo result: ')
        print(result)
        pp.pprint(result)

        # 单链接聚类
        (image_path, after_process_result) = after_process(image_path, result)
        print('linkKeams: ')
        print(after_process_result)
        # bbox dist
        after_process_ctpn(ctpn, image_path, after_process_result)
    yolo.close_session()


def test_after_process():
    image_path = r'C:\004_project\012-yolo\ZZ_Dection\test_data\201811052563102600000000000001-02M_0.jpg'
    result = \
    [ \
    [18, 'CreditCode_1', 0.9990651, [653, 753, 953, 797]], \
    [12, 'Person_1', 0.9902284, [280, 1158, 574, 1212]], \
    [6, 'Address_1', 0.9949561, [288, 1062, 567, 1117]], \
    [5, 'Company_1', 0.9971853, [287, 872, 572, 928]], \
    [0, 'BusinessLicense_1', 0.9986943, [376, 498, 1290, 686]] \
    ]

    # result = \
    # [ \
    # [18, 'BusinessLicense_1', 0.9990651, [653, 753, 953, 797]] \
    # ]

    (image_path, after_process_result) = after_process(image_path, result)

    ctpn = ctpn_predict.id_card_word_position()
    after_process_ctpn(ctpn, image_path, after_process_result)

if __name__ == '__main__':
    # test_after_process()
    detect_img_for_batch_predict()
    pass









