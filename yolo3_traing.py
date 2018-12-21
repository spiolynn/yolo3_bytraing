# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 9:38
# @Author  : panzi
# @Site    : 
# @File    : yolo3_traing.py
# @Software: PyCharm Community Edition

'''
使用yolo3模型训练自有数据
'''

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from config import YOLO3Config as C
import os
import tensorflow as tf


'''
#allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
#使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片

# per_process_gpu_memory_fraction
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config, ...)
#设置每个GPU应该拿出多少容量给进程使用，0.4代表 40%
'''

def _main():

    if C.USE_GPU:
        # 如果使用GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_id #使用 GPU 0和1
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    annotation_path = C.annotation_path
    log_dir = C.log_dir
    classes_path = C.classes_path
    anchors_path = C.anchors_path
    input_shape = C.input_shape
    weights_path = C.weights_path


    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    print('num_classes:\n' + str(class_names))
    print('anchors:\n' + str(anchors.tolist()))

    # default setting
    # 模型初始化
    # 冻结模式，1是冻结DarkNet53的层，2是冻结全部，只保留最后3层；
    is_tiny_version = len(anchors)==6
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=weights_path)
    else:
        '''
        input_shape：输入图片的尺寸，默认是(416, 416)
        anhors：默认的9种anchor box，结构是(9, 2)
        num_classes：类别个数，在创建网络时，只需类别数即可。在网络中，类别值按0~n排列，同时，输入数据的类别也是用索引表示
        load_pretrained：是否使用预训练权重。预训练权重，既可以产生更好的效果，也可以加快模型的训练速度
        freeze_body：冻结模式，1或2。其中，1是冻结DarkNet53网络中的层，2是只保留最后3个1x1的卷积层，其余层全部冻结
        weights_path：预训练权重的读取路径
        '''
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=weights_path)


    # 回调函数
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    '''reduce_lr：当评价指标不在提升时，减少学习率，每次减少10%，当验证损失值，持续3次未减少时，则终止训练。'''
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    '''early_stopping：当验证集损失值，连续增加小于0时，持续10个epoch，则终止训练'''
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


    # 训练数据准备
    val_split = C.val_split
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    # 二阶段训练
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = C.fstep_batchsize
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=C.fepoch,
                initial_epoch=0,
                callbacks=[logging, checkpoint])

        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        model.save(log_dir + 'StageOne.h5')


    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')
        batch_size = C.sstep_batchsize # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=C.fepoch + C.sepoch,
            initial_epoch=C.fepoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')
        model.save(log_dir + 'StageTwo.h5')


def get_classes(classes_path):
    classes_path = classes_path
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = anchors_path
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''

    # get a new session
    K.clear_session()
    # 在输入层中，既可显式指定图片尺寸，如(416, 416, 3)，也可隐式指定，用“?”代替，如(?, ?, 3)
    h, w = input_shape  # 尺寸
    # image_input = Input(shape=(None, None, 3))
    image_input = Input(shape=(w, h, 3))
    num_anchors = len(anchors)

    # YOLO的三种尺度，每个尺度的anchor数，类别数(这里只有一个类别)+边框4个+置信度1 = 6
    '''
    Tensor("input_2:0", shape=(?, 416/32=13, 13, 3, 6), dtype=float32)
    Tensor("input_3:0", shape=(?, 26, 26, 3, 6), dtype=float32)
    Tensor("input_4:0", shape=(?, 52, 52, 3, 6), dtype=float32)
    '''
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    ## yolo3 核心模型
    '''
    通过传入输入层image_input、每层的anchor数num_anchors//3和类别数num_classes，调用yolo_body方法，构建YOLO v3的网络model_body。其中，image_input的结构是(?, 416, 416, 3)
    在model_body中，最终的输入是image_input，最终的输出是3个矩阵的列表：
    [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    18的意思是:  每个尺度的anchor数，类别数(这里只有一个类别)+边框4个+置信度1 = 6
    3(一个尺度anchar个数)*(一个类别+边框4+1个置信度)
    '''
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('model_boby:')
    print(model_body.summary())

    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))


    if load_pretrained:
        # 加载预训练权重的逻辑块
        '''
        根据预训练权重的地址weights_path，加载权重文件，设置参数为，按名称对应by_name，略过不匹配 skip_mismatch
        skip_mismatch 重要
        '''
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

        '''
        选择冻结模式：模式1是冻结185层，模式2是保留最底部3层，其余全部冻结。整个模型共有252层；将所冻结的层，设置为不可训练，trainable=False；
        最后三层: 最底部3个1x1的卷积层，将3个特征矩阵转换为3个预测矩阵，其格式如下：
        1: (None, 13, 13, 1024) -> (None, 13, 13, 18)
        2: (None, 26, 26, 512) -> (None, 26, 26, 18)
        3: (None, 52, 52, 256) -> (None, 52, 52, 18)
        '''
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # 将loss函数也作为一层加入模型中
    '''
    Lambda是Keras的自定义层，输入为model_body.output和y_true，输出output_shape是(1,)，即一个损失值；
    自定义Lambda层的名字name为yolo_loss；
    层的参数是锚框列表anchors、类别数num_classes和IoU阈值ignore_thresh。其中，ignore_thresh用于在物体置信度损失中过滤IoU较小的框；
    yolo_loss是损失函数的核心逻辑。
    '''
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    '''
    其中，model_body.input是任意(?)个(416,416,3)的图片；y_true是已标注数据所转换的真值结构。即：
    [Tensor("input_2:0", shape=(?, 13, 13, 3, 6), dtype=float32),
    Tensor("input_3:0", shape=(?, 26, 26, 3, 6), dtype=float32),
    Tensor("input_4:0", shape=(?, 52, 52, 3, 6), dtype=float32)]
    '''
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

# 数据生成器
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                # 所有图片使用后 随机一次
                np.random.shuffle(annotation_lines)
            # 获取图片、box
            ''' image_data: (16, 416, 416, 3)  box_data: (16, 20, 5)'''
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

'''
annotation_lines：标注数据的行，每行数据包含图片路径，和框的位置信息；
batch_size：批次数，每批生成的数据个数；
input_shape：图像输入尺寸，如(416, 416)；
anchors：anchor box列表，9个宽高值；
num_classes：类别的数量；
'''
def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    # 确认参数正确
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()