

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")


import tensorflow as tf
import functools
import json
from util import *

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
    def __init__(self, input_tensor, config=None):
        self.forward
        self.backward
        self.input = input_tensor
        self.config = config


    @define_scope
    def forward(self):
        # Conv-Box 1
        # box1 = self.config['Conv-Box-1']
        #
        # conv1 = tf.layers.conv2d(
        #     inputs = self.input,
        #     filters = box1['filters'],
        #     kernel_size = box1['kernel_size'],
        #     padding = "same",
        #     activation = mapActivationFunc(box1['activation']))
        #
        # pool1 = tf.layers.max_pooling2d(
        #     inputs=conv1,
        #     pool_size = box1['pool_size'],
        #     strides= = box1['strides'])
        #
        #
        # # Conv-Box-2
        # box2 = self.config['Conv-Box-2']
        # conv2 = tf.layers.conv2d(
        #     inputs = self.input,
        #     filters = box2['filters'],
        #     kernel_size = box2['kernel_size'],
        #     padding = "same",
        #     activation = mapActivationFunc(box2['activation']))
        #
        # pool2 = tf.layers.max_pooling2d(
        #     inputs=conv2,
        #     pool_size = box2['pool_size'],
        #     strides= = box2['strides'])

        #Dense layer
        return "done"




def main():
    obj = CNN()
    print obj.forward
    # print obj.forward()

main()
