from __future__ import division
import tensorflow as tf
from config import ModelConfig
import numpy as np


class Model:

    def __init__(self, num_hidden_layers=3, hidden_size=1024):
        # Placeholders
        self.x_mixed = tf.placeholder(dtype=tf.float32, shape=(None, ModelConfig.N_FFT // 2 + 1), name='x_mixed')
        self.y_src1 = tf.placeholder(dtype=tf.float32, shape=(None, ModelConfig.N_FFT // 2 + 1), name='y_src1')
        self.y_src2 = tf.placeholder(dtype=tf.float32, shape=(None, ModelConfig.N_FFT // 2 + 1), name='y_src2')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        # Networks
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        #################################################
        #                  DNN Structure                #
        #################################################
        self.h1_out = tf.nn.dropout(
            tf.layers.dense(inputs=self.x_mixed, units=self.hidden_size, activation=tf.nn.sigmoid, name='h1_out'),
            keep_prob=self.keep_prob)
        self.h2_out = tf.nn.dropout(
            tf.layers.dense(inputs=self.h1_out, units=self.hidden_size, activation=tf.nn.sigmoid, name='h2_out'),
            keep_prob=self.keep_prob)
        self.h3_out = tf.nn.dropout(
            tf.layers.dense(inputs=self.h2_out, units=self.hidden_size, activation=tf.nn.sigmoid, name='h3_out'),
            keep_prob=self.keep_prob)

        self.output_size = ModelConfig.N_FFT // 2 + 1
        self.y_hat_src1 = tf.layers.dense(inputs=self.h3_out,
                                          units=self.output_size,
                                          activation=tf.nn.sigmoid,
                                          name='y_hat_src1')
        self.y_hat_src2 = tf.layers.dense(inputs=self.h3_out,
                                          units=self.output_size,
                                          activation=tf.nn.sigmoid,
                                          name='y_hat_src2')

        #################################################
        #                  Time Masking                 #
        #################################################
        self.y_tilde_src1 = np.abs(self.y_hat_src1) / (np.abs(self.y_hat_src1) + np.abs(self.y_hat_src2) + np.finfo(float).eps) * self.x_mixed
        self.y_tilde_src2 = np.abs(self.y_hat_src2) / (np.abs(self.y_hat_src1) + np.abs(self.y_hat_src2) + np.finfo(float).eps) * self.x_mixed

        #################################################
        #                 Loss & Optimizer              #
        #################################################
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        if ModelConfig.LOSS == 'L2':
            self.loss \
                = tf.reduce_mean(tf.square(self.y_src1 - self.y_tilde_src1) + \
                                 tf.square(self.y_src2 - self.y_tilde_src2), name='loss')
        elif ModelConfig.LOSS == 'L1':
            self.loss \
                = tf.reduce_mean(tf.abs(self.y_src1 - self.y_tilde_src1) + \
                                 tf.abs(self.y_src2 - self.y_tilde_src2), name='loss')
        
        self.optimizer \
            = tf.train.RMSPropOptimizer(learning_rate=ModelConfig.LR).minimize(loss=self.loss, global_step=self.global_step)

    def train(self, sess, summary_op, feed_dict):
        loss, _, summary = sess.run([self.loss, self.optimizer, summary_op], feed_dict=feed_dict)
        return loss, summary

    def predict(self, sess, feed_dict):
        pred_y_src1, pred_y_src2 = sess.run([self.y_tilde_src1, self.y_tilde_src2], feed_dict=feed_dict)
        return pred_y_src1, pred_y_src2

