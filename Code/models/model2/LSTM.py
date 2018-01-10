import tensorflow as tf
import numpy as np

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")

import functools
import json
from util import *


def define_scope(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope('rnn_'+function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class LSTM:
    def __init__(self,input_tesor, config, is_training=True):
        self.config = config
        self.input_tesor = input_tesor
        self.is_training = is_training
        self.forward

    @define_scope
    def forward(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.config["hidden_dim"],state_is_tuple=True, initializer=tf.truncated_normal_initializer(stddev=0.02))
        output,state = tf.nn.dynamic_rnn(cell,self.input_tesor,dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1) #Gather takes two values param and indices. works like slicing
        dense_config = self.config['Dense']
        logits = tf.layers.dense(inputs=last,units=dense_config['units_layer_1'],activation=mapActivationFunc(dense_config['activation']),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        dropout_1 = tf.layers.dropout(inputs=logits, rate=dense_config['dropout_1_rate'], training=self.is_training)
        batch_norm_dense_1 = tf.contrib.layers.batch_norm(inputs=dropout_1)
        dense_2 =  tf.layers.dense(inputs=batch_norm_dense_1, units= self.config['vocab_size'], activation=mapActivationFunc(dense_config['activation']),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return dense_2

#example for lstm class
def main():
    config2 = {"INPUT_DIMENSION":100,
    "SEQUENCE_LENGTH":29,
    "VOCABULARY_SIZE":10,
    "HIDDEN_DIMENSTION":20,
    "NO_EXAMPLES":1000,
    "TRAIN_EXAMPLES":500,
    "BATCH_SIZE":500}
    train_input = np.random.rand(config2["NO_EXAMPLES"],config2["SEQUENCE_LENGTH"],config2["INPUT_DIMENSION"])
    randomNumbers = np.random.randint(config2["VOCABULARY_SIZE"],size=config2["NO_EXAMPLES"])
    reshapedRandomNumbers = randomNumbers.reshape(-1)
    train_output = np.eye(config2["VOCABULARY_SIZE"])[reshapedRandomNumbers]

    test_input = train_input[config2["TRAIN_EXAMPLES"]:]
    test_output = train_output[config2["TRAIN_EXAMPLES"]:]

    train_input = train_input[:config2["TRAIN_EXAMPLES"]]
    train_output = train_output[:config2["TRAIN_EXAMPLES"]]

    config = loadConfig('config.json')

    data = tf.placeholder(tf.float32,[None,config2["SEQUENCE_LENGTH"],config2["INPUT_DIMENSION"]])
    target = tf.placeholder(tf.float32,[None,config2["VOCABULARY_SIZE"]])
    lstm = LSTM(data, config['LSTM'])

    prediction = lstm.forward

    loss = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(loss)

    batch_size = 64
    total = len(train_input)
    epochs = 100
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

        for e in range(epochs):
            for i in range(0, total, batch_size):
                inp,out = train_input[i:i+batch_size],train_output[i:i+batch_size]
                loss_out = sess.run(loss, {data:inp, target:out})
                prediction = sess.run(minimize,{data:inp, target:out})
            print("Epoch - "+str(loss_out))



# main()
