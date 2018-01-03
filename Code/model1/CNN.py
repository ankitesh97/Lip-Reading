
import tensorflow as tf
import functools
import json

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
        box1 = self.config['Conv-Box-1']
        conv1 = tf.layers.conv2d(
            inputs = self.input,
            filters = box1['filters'],
            kernel_size = box1['kernel_size'],
            padding = "same",
            
        )





def main():
    obj = CNN()
    print obj.forward
    # print obj.forward()

main()
