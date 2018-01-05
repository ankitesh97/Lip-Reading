

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")


import tensorflow as tf
import functools
import json
from util import *
import numpy as np

def define_scope(function):
    attribute = '_cache_' + function.__name__
    print attribute
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope('cnn_'+function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class CNN:
    def __init__(self, input_tensor, config=None, is_training):
        self.forward
        self.backward
        self.input = input_tensor
        self.config = config
        wdwqdw
        self.is_training = is_training

    @define_scope
    def forward(self):
        # Conv-Box 1
        box1 = self.config['Conv-Box-1']

        conv1 = tf.layers.conv2d(
            inputs = self.input,
            filters = box1['filters'],
            kernel_size = box1['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box1['activation']))

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size = box1['pool_size'],
            strides= = box1['strides'])


        # Conv-Box-2
        box2 = self.config['Conv-Box-2']
        conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = box2['filters'],
            kernel_size = box2['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box2['activation']))

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size = box2['pool_size'],
            strides= = box2['strides'])


        # Conv-Box-3
        box3 = self.config['Conv-Box-3']
        conv3 = tf.layers.conv2d(
            inputs = pool2,
            filters = box3['filters'],
            kernel_size = box3['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box3['activation']))


        # Conv-Box-4
        box4 = self.config['Conv-Box-4']
        conv4 = tf.layers.conv2d(
            inputs = conv3,
            filters = box4['filters'],
            kernel_size = box4['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box4['activation']))

        pool4 = tf.layers.max_pooling2d(
            inputs=conv4,
            pool_size = box4['pool_size'],
            strides= = box4['strides'])


        # Conv-Box-5
        box5 = self.config['Conv-Box-5']
        conv5 = tf.layers.conv2d(
            inputs = pool4,
            filters = box5['filters'],
            kernel_size = box5['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box5['activation']))

        pool5 = tf.layers.conv2d(
            inputs=conv5,
            pool_size = box5['pool_size'],
            strides= box5['strides']
        )

        output_dim = ()
        pool5_flat =  tf.reshape(pool5, [-1, np.product(output_dim)])

        #dense layer 1
        dense_config = self.config["Dense"]
        dense_1 =  tf.layers.dense(inputs=pool5_flat, units= dense_config['units_layer_1'], activation=mapActivationFunc[dense_config['activation']])
        dropout_1 = tf.layers.dropout(inputs=dense_1, rate=dense_config['dropout_1_rate'], training=is_training)
        batch_norm_dense_1 = tf.contrib.layers(inputs=dropout_1)

        #dense layer 2
        dense_2 =  tf.layers.dense(inputs=batch_norm_dense_1, units= dense_config['units_layer_2'], activation=mapActivationFunc[dense_config['activation']])
        dropout_2 = tf.layers.dropout(inputs=dense_2, rate=dense_config['dropout_2_rate'], training=is_training)
        batch_norm_dense_2 = tf.contrib.layers(inputs=dropout_2)

        #output layer
        features =  tf.layers.dense(inputs=batch_norm_dense_2, units= dense_config['feature_dim'])

        return features

def main():
    obj = CNN()
    print obj.forward
    # print obj.forward()

main()
