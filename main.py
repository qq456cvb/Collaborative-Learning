#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-preact18-mixup.py
# Author: Tao Hu <taohu620@gmail.com>,  Yauheni Selivonchyk <y.selivonchyk@gmail.com>
# Adapted by Neil You

import argparse
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *

BATCH_SIZE = 64
CLASS_NUM = 10

LR_SCHEDULE = [(0, 0.1), (150, 0.01), (225, 0.001), (290, 0.0001)]
WEIGHT_DECAY = 1e-4


T = 2
beta = 0.5


def densenet_block(input, num_layers, growth, keep_prob):
    crt = input
    tmp = None
    for i in range(num_layers):
        with tf.variable_scope('layer%d' % i):
            tmp = BNReLU(crt)
            tmp = Conv2D('conv%d' % i, tmp, growth, kernel_size=3, strides=1, use_bias=False)
            tmp = Dropout('dropout%d' % i, tmp, keep_prob=keep_prob)
            crt = tf.concat((crt, tmp), axis=1)
    return tmp


def recursive_split_block(input, block_fun, splits, depth):
    @tf.custom_gradient
    def scale_grad_layer(x):
        def grad(dy):
            return dy / splits

        return tf.identity(x), grad
    outputs = []
    if depth == 0:
        return [input]
    for i in range(splits):
        with tf.variable_scope('block%d' % i):
            child = scale_grad_layer(input)
            child = block_fun(child)
            outputs.extend(recursive_split_block(child, block_fun, splits, depth - 1))
    return outputs


class DenseNet_Cifar(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label')]

    def build_graph(self, image, label):
        assert tf.test.is_gpu_available()

        depth = 40
        num_layers_total = depth - 4
        num_blocks = 3

        MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE
        image = tf.transpose(image, [0, 3, 1, 2])

        pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
        with argscope([Conv2D, BatchNorm, GlobalAvgPooling, AvgPooling], data_format='channels_first'), \
                argscope(Conv2D, kernel_initializer=pytorch_default_init):
            net = Conv2D('pre_conv', image, 16, kernel_size=3, strides=1, use_bias=False)

            net = densenet_block(net, num_layers_total // num_blocks, 12, 0.8)
            net = BNReLU(net)

            def block_fun(x):
                x = Conv2D('conv_trans', x, x.shape[1], kernel_size=1, strides=1, use_bias=False)
                x = Dropout('dropout_trans', x, keep_prob=0.8)
                x = AvgPooling('avgpool_trans', x, 2)
                x = densenet_block(x, num_layers_total // num_blocks, 12, 0.8)
                x = BNReLU(x)
                return x

            heads = recursive_split_block(net, block_fun, 2, 2)

            costs = []

            heads_logits = []
            for i, head in enumerate(heads):
                with tf.variable_scope('head%d' % i):
                    net = GlobalAvgPooling('gap', head)
                    logits = FullyConnected('linear', net, CLASS_NUM,
                                            kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
                    heads_logits.append(logits)

            for i, logits in enumerate(heads_logits):
                hard_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
                q = tf.nn.softmax((tf.add_n(heads_logits) - logits) / (len(heads_logits) - 1) / T)
                # TODO: fix target q?
                q = tf.stop_gradient(q)
                soft_loss = -tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(logits / T), axis=-1))
                loss = beta * hard_loss + (1 - beta) * soft_loss
                costs.append(loss)

        total_loss = tf.add_n(costs, name='loss')
        single_label = tf.cast(tf.argmax(label, axis=1), tf.int32)
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(heads_logits[0], single_label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), total_loss)
        add_param_summary(('.*/W', ['histogram']))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')

        return tf.add_n([total_loss, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
        ]
        ds = AugmentImageComponent(ds, augmentors)

    batch = BATCH_SIZE
    ds = BatchData(ds, batch, remainder=not isTrain)

    def f(dp):
        images, labels = dp
        one_hot_labels = np.eye(CLASS_NUM)[labels]  # one hot coding
        return [images, one_hot_labels]

    ds = MapData(ds, f)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_folder = 'train_log'
    logger.set_logger_dir(os.path.join(log_folder))

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=DenseNet_Cifar(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate', LR_SCHEDULE)
        ],
        max_epoch=300,
        steps_per_epoch=len(dataset_train),
        session_init=SaverRestore(args.load) if args.load else None
    )
    launch_train_with_config(config, SimpleTrainer())
