

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
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope('cnn_'+function.__name__):
                setattr(self, attribute, function(self))

        return getattr(self, attribute)

    return decorator


class CNN:
    def __init__(self, input_tensor, config, is_training=True):
        self.config = config
        self.input_tensor = input_tensor
        self.is_training = is_training
        self.forward


    @define_scope
    def forward(self):
        # Conv-Box 1
        box1 = self.config['Conv-Box-1']
        conv1 = tf.layers.conv2d(
            inputs = self.input_tensor,
            filters = box1['filters'],
            kernel_size = box1['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box1['activation']))

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size = box1['pool_size'],
            strides = box1['strides'])


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
            strides = box2['strides'])


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
            strides = box4['strides'])


        # Conv-Box-5
        box5 = self.config['Conv-Box-5']
        conv5 = tf.layers.conv2d(
            inputs = pool4,
            filters = box5['filters'],
            kernel_size = box5['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box5['activation']))

        pool5 = tf.layers.max_pooling2d(
            inputs=conv5,
            pool_size = box5['pool_size'],
            strides= box5['strides']
        )

        output_dim = pool5.get_shape()[1:]

        pool5_flat =  tf.reshape(pool5, [-1, np.product(output_dim)])

        #dense layer 1
        dense_config = self.config["Dense"]
        features =  tf.layers.dense(inputs=pool5_flat, units= dense_config['feature_dim'])
        return features



#this function is to test the class
def main():

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    total = train_data.shape[0]
    batch_size = 64
    config = loadConfig('config.json')
    inp_shape = config['CNN']['Input_shape']
    input_te = tf.placeholder(tf.float32, shape=[None,28*28])
    input_tensor = tf.reshape(input_te, [-1,28,28,1])
    cnn_object = CNN(input_tensor,config["CNN"])
    forward_op = cnn_object.forward
    prdict = tf.nn.softmax(forward_op, name="prediction")
    probab = tf.nn.softmax(forward_op)
    onehot_labels = tf.placeholder(tf.int32, shape=[None, 10])
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=forward_op)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step()
    )
    n_epochs=4
    nb_classes=10
    init = tf.global_variables_initializer()
    print(total)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        for epoch in range(n_epochs):
            zipped = zip(train_data,train_labels)
            X,y = zip(*zipped)
            X = np.array(X)
            y = np.array(y)
            for i in range(0, total, batch_size):
                x_curr_batch = X[i:i+batch_size]
                y_curr = y[i:i+batch_size].reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y_curr]
                loss_val = sess.run(loss, feed_dict={input_te:x_curr_batch, onehot_labels: one_hot_targets})
                sess.run(train_op, feed_dict={input_te:x_curr_batch, onehot_labels: one_hot_targets})

            print("epoch: "+str(epoch)+" loss: "+str(loss_val))
            saver.save(sess, './trained_models/clstmModel', global_step=epoch,write_meta_graph=False, max_to_keep=4)

        saver.save(sess, "./trained_models/clstmModel_final")




# main()
