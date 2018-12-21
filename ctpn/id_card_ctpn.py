'''
id_card_ctpn
'''

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.recurrent import GRU
from keras.layers.core import Reshape, Dense, Flatten, Permute, Lambda, Activation
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import regularizers
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import ctpn.utils as utils
from ctpn.text_connect_cfg import Config as TextLineCfg
import os
import time
import cv2
import matplotlib.pyplot as plt
import ctpn.utils
from ctpn.text_proposal_connector_oriented import TextProposalConnectorOriented
import numpy as np

BBOX_THRESH = 0.8
NMSBOX_THRESH = 0.15
DEBUG = True

class id_card_ctpn(object):

    def __init__(self):
        pass

    def bulid_model(self,vgg_path):
        inp, nn = self.nn_base((None, None, 3), trainable=False,vgg_path=vgg_path)
        cls, regr, cls_prod = self.rpn(nn)
        self.basemodel = Model(inp, [cls, regr, cls_prod])

    def load_model(self,model_path):
        self.basemodel.load_weights(model_path)

    def predict(self,img):
        img, scale = self.resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        h, w, c = img.shape
        m_img = img - utils.IMAGE_MEAN
        m_img = np.expand_dims(m_img,axis=0)
        cls,regr,cls_prod = self.basemodel.predict(m_img)

        anchor = utils.gen_anchor((int(h/16),int(w/16)),16)

        bbox = utils.bbox_transfor_inv(anchor, regr)
        bbox = utils.clip_box(bbox, [h, w])

        #score > 0.8
        fg = np.where(cls_prod[0,:,1]>BBOX_THRESH)[0]
        select_anchor = bbox[fg,:]
        select_score = cls_prod[0,fg,1]
        select_anchor = select_anchor.astype('int32')
        keep_index = utils.filter_bbox(select_anchor, 16)

        #nms
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score,(select_score.shape[0],1))
        nmsbox = np.hstack((select_anchor,select_score))
        keep = utils.nms(nmsbox,NMSBOX_THRESH)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])

        if DEBUG:
            m_img_copy = img.copy()
            for i in range(select_anchor.shape[0]):
                box = select_anchor[i]
                cv2.rectangle(m_img_copy, (box[0], box[1]),
                              (box[2], box[3]), (255, 0 , 0), 2)

            cv2.imwrite('1.jpg', m_img_copy)

        text = text.astype('int32')

        # t = int(time.time())

        return img,text,scale

    def rpn_loss_regr(self,y_true, y_pred):
        """
        smooth L1 loss
        y_ture [1][HXWX9][3] (class,regr)
        y_pred [1][HXWX9][2] (reger)
        """
        sigma = 9.0
        cls = y_true[0, :, 0]
        regr = y_true[0, :, 1:3]
        regr_keep = tf.where(K.equal(cls, 1))[:, 0]
        regr_true = tf.gather(regr, regr_keep)
        regr_pred = tf.gather(y_pred[0], regr_keep)
        diff = tf.abs(regr_true - regr_pred)
        less_one = tf.cast(tf.less(diff, 1.0 / sigma), 'float32')
        loss = less_one * 0.5 * diff ** 2 * sigma + tf.abs(1 - less_one) * (diff - 0.5 / sigma)
        loss = K.sum(loss, axis=1)
        return K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.0))


    def rpn_loss_cls(self,y_true, y_pred):
        """
        softmax loss
        y_true [1][1][HXWX9] class
        y_pred [1][HXWX9][2] class
        """
        y_true = y_true[0][0]
        cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
        cls_true = tf.gather(y_true, cls_keep)
        cls_pred = tf.gather(y_pred[0], cls_keep)
        cls_true = tf.cast(cls_true, 'int64')
        # loss = K.sparse_categorical_crossentropy(cls_true,cls_pred,from_logits=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
        return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


    def nn_base(self,input,trainable,vgg_path):
        base_model = VGG16(weights=None, include_top=False, input_shape=input)
        base_model.load_weights(vgg_path)
        if (trainable == False):
            for ly in base_model.layers:
                ly.trainable = False
        return base_model.input,base_model.get_layer('block5_conv3').output


    def reshape(self,x):
        b = tf.shape(x)
        x = tf.reshape(x, [b[0] * b[1], b[2], b[3]])
        return x


    def reshape2(self,x):
        x1, x2 = x
        b = tf.shape(x2)
        x = tf.reshape(x1, [b[0], b[1], b[2], 256])
        return x


    def reshape3(self,x):
        b = tf.shape(x)
        x = tf.reshape(x, [b[0], b[1] * b[2] * 10, 2])
        return x


    def rpn(self,base_layers):
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                   name='rpn_conv1')(base_layers)
        x1 = Lambda(self.reshape, output_shape=(None, 512))(x)
        x2 = Bidirectional(GRU(128, return_sequences=True), name='blstm')(x1)
        x3 = Lambda(self.reshape2,output_shape=(None, None, 256))([x2, x])
        x3 = Conv2D(512, (1, 1), padding='same', activation='relu', name='lstm_fc')(x3)
        cls = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_class')(x3)
        regr = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_regress')(x3)
        cls = Lambda(self.reshape3, output_shape=(None, 2), name='rpn_class_reshape')(cls)
        cls_prod = Activation('softmax', name='rpn_cls_softmax')(cls)
        regr = Lambda(self.reshape3, output_shape=(None, 2), name='rpn_regress_reshape')(regr)
        return cls, regr, cls_prod

    def resize_im(self,im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f