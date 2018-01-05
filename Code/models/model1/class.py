# ef load_src(name, fpath):
#     import os, imp
#     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
#
# load_src("util", "../../utils/model1.py")
#

import tensorflow as tf
import functools
import json
# from util import *
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

class LSTM:
    def __init__(self,config=None):
        self.config = config
        self.forward
        # self.input = input_tensor

    @define_scope
    def forward(self):
        data = tf.placeholder(tf.float32,[None,self.config["SEQUENCE_LENGTH"],self.config["INPUT_DIMENSION"]])
        target = tf.placeholder(tf.float32,[None,self.config["VOCABULARY_SIZE"]])
        cell = tf.nn.rnn_cell.LSTMCell(self.config["HIDDEN_DIMENSTION"],state_is_tuple=True)
        output,state = tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1) #Gather takes two values param and indices. works like slicing
        weight = tf.Variable(tf.truncated_normal([self.config["HIDDEN_DIMENSTION"],int(target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1,shape=[target.get_shape()[1]]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction,target,data

def main():
    config = {"INPUT_DIMENSION":100,
    "SEQUENCE_LENGTH":29,
    "VOCABULARY_SIZE":10,
    "HIDDEN_DIMENSTION":200,
    "NO_EXAMPLES":1000,
    "TRAIN_EXAMPLES":500,
    "BATCH_SIZE":500}
    train_input = np.random.rand(config["NO_EXAMPLES"],config["SEQUENCE_LENGTH"],config["INPUT_DIMENSION"])
    randomNumbers = np.random.randint(config["VOCABULARY_SIZE"],size=config["NO_EXAMPLES"])
    reshapedRandomNumbers = randomNumbers.reshape(-1)
    train_output = np.eye(config["VOCABULARY_SIZE"])[reshapedRandomNumbers]

    test_input = train_input[config["TRAIN_EXAMPLES"]:]
    test_output = train_output[config["TRAIN_EXAMPLES"]:]

    train_input = train_input[:config["TRAIN_EXAMPLES"]]
    train_output = train_output[:config["TRAIN_EXAMPLES"]]

    lstm = LSTM(config)
    prediction,target,data = lstm.forward
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)
    batch_size = 500
    no_of_batches = int(len(train_input)/batch_size)
    epoch = 10
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp,out = train_input[ptr:ptr+batch_size],train_output[ptr:ptr+batch_size]
            ptr+=batch_size
            prediction = sess.run(minimize,{data:inp ,target:out})
        print "Epoch -",str(i)
    mistakes = tf.not_equal(tf.argmax(target,1),tf.argmax(prediction,1))
    error = tf.reduce_mean(tf.cast(mistakes,tf.float32))
    incorrect = sess.run(error,{data:test_input, target:test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()

main()
