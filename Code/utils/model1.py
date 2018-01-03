
import tensorflow as tf

def mapActivationFunc(activation_name):
    if(activation_name=='relu'):
        return tf.nn.relu
