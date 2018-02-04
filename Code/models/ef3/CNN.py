

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "./loadData.py")


import tensorflow as tf
import functools
import json
from util import *
import numpy as np
import sys
from loadData import *


CONFIG_FILE = 'config_prod.json'
dynamic_config = loadConfig(CONFIG_FILE)
train_flag = dynamic_config['Training']['is_training']
model_save_add = dynamic_config["Training"]['save_file_address']

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
            activation = mapActivationFunc(box1['activation']),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

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
            activation = mapActivationFunc(box2['activation']),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

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
            activation = mapActivationFunc(box3['activation']),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.01))


        # Conv-Box-4
        box4 = self.config['Conv-Box-4']
        conv4 = tf.layers.conv2d(
            inputs = conv3,
            filters = box4['filters'],
            kernel_size = box4['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box4['activation']),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

        # Conv-Box-5
        box5 = self.config['Conv-Box-5']
        conv5 = tf.layers.conv2d(
            inputs = conv4,
            filters = box5['filters'],
            kernel_size = box5['kernel_size'],
            padding = "same",
            activation = mapActivationFunc(box5['activation']),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

        pool5 = tf.layers.max_pooling2d(
            inputs=conv5,
            pool_size = box5['pool_size'],
            strides= box5['strides']
        )

        output_dim = pool5.get_shape()[1:]

        pool5_flat =  tf.reshape(pool5, [-1, np.product(output_dim)], name="cnn_final_layer")

        #dense layer 1
        dense_config = self.config["Dense"]
        dense_1 =  tf.layers.dense(inputs=pool5_flat, units= dense_config['units_layer_1'], activation=mapActivationFunc(dense_config['activation']),kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        dropout_1 = tf.layers.dropout(inputs=dense_1, rate=dense_config['dropout_1_rate'], training=self.is_training)
        batch_norm_dense_1 = tf.contrib.layers.batch_norm(inputs=dropout_1)

        #dense layer 2
        dense_2 =  tf.layers.dense(inputs=batch_norm_dense_1, units= dense_config['units_layer_2'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        dropout_2 = tf.layers.dropout(inputs=dense_2, rate=dense_config['dropout_2_rate'], training=self.is_training)
        batch_norm_dense_2 = tf.contrib.layers.batch_norm(inputs=dropout_2)

        #output layer
        features =  tf.layers.dense(inputs=batch_norm_dense_2, units= dense_config['feature_dim'], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

        return features

def train(onehot_labels, predicted, learning_rate):
    # epsilon = tf.constant(value=0.00001)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=predicted))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    regularized_loss = 0.005 * sum(tf.nn.l2_loss(tf_var)
    for tf_var in tf.trainable_variables()
    if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
    loss += regularized_loss
    minimize = optimizer.minimize(loss)

    return minimize, loss


#this function is to test the class
def main():

    config = loadConfig('config_prod.json')
    inp_shape = config['CNN']['Input_shape']
    input_te = tf.placeholder(tf.float32, shape=[None,72,42,29])
    is_train = tf.placeholder(tf.bool, name="is_train")

    training_params = config["Training"]

    time_steps = config['LSTM']['time_steps']
    batch_size = training_params['batch_size']

    cnn_object = CNN(input_te,config["CNN"])
    forward_op = cnn_object.forward
    predict = tf.nn.softmax(forward_op, name="prediction")
    nb_classes = training_params['nb_classes']
    onehot_labels = tf.placeholder(tf.int32, shape=[None, nb_classes])
    # learning_rate = training_params['learning_rate']
    learning_rate_init = tf.constant(training_params['learning_rate'], dtype=tf.float32)
    learning_rate = tf.Variable(training_params['learning_rate'],trainable=False, dtype=tf.float32)

    global_step = tf.Variable(0, trainable=False)
    global_step_increment = tf.assign(global_step, global_step+1)
    one = tf.constant(1,dtype=tf.float32)
    decay_const = tf.constant(1,dtype=tf.float32)
    decay_op = tf.multiply(learning_rate_init, tf.divide(one,tf.add(one,tf.multiply(decay_const,tf.to_float(global_step)))))
    learning_rate_decay = tf.assign(learning_rate,decay_op)
    train_op, loss = train(onehot_labels, forward_op, learning_rate)
    classes = tf.argmax(predict, axis=1)

    init = tf.global_variables_initializer()
    Losses = []
    Losses_train = []
    saver = tf.train.Saver(max_to_keep=8)
    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    acc_size = training_params['acc_size']

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )
    with tf.Session(config=config) as sess:

        max_acc = 0
        sess.run(init)
        print "---------------Starting to train-------------"
        sys.stdout.flush()
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

        for e in range(epochs):
            total = loadDataQueue(is_colored)
            if(e%3==0):
                with open("control.txt",'r') as f:
                    control = f.read()
                    if control.strip() == "1":
                       print "-----------------stopping the training process .........."
                       sys.stdout.flush()
                       break
            batch_count = 0
            for i in range(0,total,batch_size):
                X, y, sequence_length = getNextBatch(batch_size) #will return time x batch x feature (24)
                # X = np.swapaxes(X, 0, 1)
                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={input_te:X,onehot_labels:one_hot_targets, is_train:True})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()

            emptyDataQueue()
            total_val = loadDataQueue(data='val')

            X_acc,Y, sequence_length = getNextBatch(total_val)
            # X_acc = np.swapaxes(X_acc, 0, 1)
            Y = Y.reshape(-1)
            one_hot_targets = np.eye(nb_classes)[Y]

            emptyDataQueue()
            loadDataQueue()
            X_acc_train,Y_train,sequence_length_train = getNextBatch(total_val)
            # X_acc_train = np.swapaxes(X_acc_train, 0, 1)
            Y_train = Y_train.reshape(-1)
            one_hot_targets_train = np.eye(nb_classes)[Y_train]

            predicted_classes = sess.run(classes, feed_dict={input_te:X_acc, onehot_labels:one_hot_targets,is_train:False})
            loss_val = sess.run(loss,feed_dict={input_te:X_acc, onehot_labels:one_hot_targets,is_train:False})

            predicted_classes_train = sess.run(classes, feed_dict={input_te:X_acc_train, onehot_labels:one_hot_targets_train,is_train:False})
            loss_val_train = sess.run(loss,feed_dict={input_te:X_acc_train, onehot_labels:one_hot_targets_train,is_train:False})
            Losses.append(loss_val)
            Losses_train.append(loss_val_train)
            count=0

            for i in range(len(Y)):
                if predicted_classes[i] == Y[i]:
                    count = count + 1

            acc_val = (count*(1.0)/len(predicted_classes)) *100

            count = 0
            for i in range(len(Y_train)):
                if predicted_classes_train[i] == Y_train[i]:
                    count = count + 1
            acc_train = (count*(1.0)/len(predicted_classes_train)) *100

            if acc_val > max_acc:
                max_acc = acc_val
                saver.save(sess, model_save_add+'model_max_acc')


            emptyDataQueue()

            print "Epoch: "+str(e)+" Loss_train: "+str(loss_val_train)+" Train Accuracy: "+str(acc_train)+"%"+ " Loss_val: "+str(loss_val)+" val Accuracy: "+str(acc_val)+"%"
            sys.stdout.flush()
            sess.run(global_step_increment)
            sess.run(learning_rate_decay)
    #
        saver.save(sess, model_save_add+"model_final")
        print max_acc
        # print graph.get_tensor_by_name('rnn_forward/dense_1/bias:0').eval()
        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses),"losses_train":map(float,Losses_train)}))




main()
