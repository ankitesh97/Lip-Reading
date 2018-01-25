import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from  tensorflow.python.framework import tensor_shape
# _state_size_with_prefix = tf.contrib.nn._state_size_with_prefix

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

def _state_size_with_prefix(state_size, prefix=None):
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be a list.")
            result_state_size = prefix + result_state_size
    return result_state_size

def get_initial_cell_state(cell, initializer, batch_size, dtype,curr_bc):

    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i,curr_bc)
                for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None,curr_bc)

    return init_state

def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index,curr_bc):
        args = kwargs.copy()
        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype
        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([curr_bc] + [1] * len(shape)))
        return var

    return variable_state_initializer


def make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
    def gaussian_state_initializer(shape, batch_size, dtype, index,curr_bc):
        init_state = initializer(shape, batch_size, dtype, index, curr_bc)
        if deterministic_tensor is not None:
            return tf.cond(deterministic_tensor,
                lambda: init_state,
                lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev))
        else:
            return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev)
    return gaussian_state_initializer

class LSTM:
    def __init__(self,input_tesor, config, seq_len, is_training=True):
        self.config = config
        self.input_tesor = input_tesor
        self.is_training = is_training
        self.seq_len = seq_len
        self.forward

    @define_scope
    def forward(self):
        celllstm = tf.nn.rnn_cell.LSTMCell(self.config["hidden_dim"],state_is_tuple=True, initializer=tf.truncated_normal_initializer(stddev=0.02))
        cellgru = tf.nn.rnn_cell.GRUCell(self.config["hidden_dim"], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        cellBatchnormLstm = tf.contrib.rnn.LayerNormBasicLSTMCell(self.config["hidden_dim"])
        deterministic = tf.constant(False)
        initializer =  make_gaussian_state_initializer(make_variable_state_initializer(),deterministic)
        # init_state = get_initial_cell_state(celllstm, initializer, self.config["batch_size"] ,tf.float32,curr_bc=tf.shape(self.input_tesor)[0])
        # print init_state
        # initializer_pickes = tf.gather(init_state,tf.range(0, tf.shape(self.input_tesor)[0]))
        # print initializer_pickes
        output,_ = tf.nn.dynamic_rnn(celllstm,self.input_tesor,sequence_length=self.seq_len, dtype=tf.float32)
        if self.seq_len is None:
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1) #Gather takes two values param and indices. works like slicing
        else:
            batch_size = tf.shape(output)[0]
            print batch_size
            max_length = int(output.get_shape()[1])
            output_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (self.seq_len - 1)
            flat = tf.reshape(output, [-1, output_size])
            last = tf.gather(flat, index)
        dense_config = self.config['Dense']
        logits = tf.layers.dense(inputs=last,units=dense_config['units_layer_1'],activation=mapActivationFunc(dense_config['activation']),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        dropout_1 = tf.layers.dropout(inputs=logits, rate=dense_config['dropout_1_rate'], training=self.is_training)
        batch_norm_dense_1 =  tf.contrib.layers.batch_norm(inputs=dropout_1)
        dense_2 =  tf.layers.dense(inputs=batch_norm_dense_1, units= dense_config['units_layer_2'], activation=mapActivationFunc(dense_config['activation']),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

        batch_norm_dense_2 = tf.contrib.layers.batch_norm(inputs=dense_2)

        dense_3 =  tf.layers.dense(inputs=batch_norm_dense_2, units= self.config['vocab_size'], activation=mapActivationFunc(dense_config['activation']),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return dense_3

#example for lstm class
def main():
    config2 = {"INPUT_DIMENSION":10,
    "SEQUENCE_LENGTH":12,
    "VOCABULARY_SIZE":10,
    "HIDDEN_DIMENSTION":20,
    "NO_EXAMPLES":4,
    "TRAIN_EXAMPLES":4,
    "BATCH_SIZE":2}
    train_input = np.random.rand(config2["NO_EXAMPLES"],config2["SEQUENCE_LENGTH"],config2["INPUT_DIMENSION"])
    randomNumbers = np.random.randint(config2["VOCABULARY_SIZE"],size=config2["NO_EXAMPLES"])
    reshapedRandomNumbers = randomNumbers.reshape(-1)
    train_output = np.eye(config2["VOCABULARY_SIZE"])[reshapedRandomNumbers]
    train_input[0][10:] = 0
    train_input[1][8:] = 0
    #
    test_input = train_input[config2["TRAIN_EXAMPLES"]:]
    test_output = train_output[config2["TRAIN_EXAMPLES"]:]
    #
    # train_input = train_input[:config2["TRAIN_EXAMPLES"]]
    # train_output = train_output[:config2["TRAIN_EXAMPLES"]]
    #
    config = loadConfig('config.json')
    #
    data = tf.placeholder(tf.float32,[None,config2["SEQUENCE_LENGTH"],config2["INPUT_DIMENSION"]])
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    l2 = tf.reduce_sum(used, reduction_indices=1)
    l = tf.cast(l2, tf.int32)

    target = tf.placeholder(tf.float32,[None,config2["VOCABULARY_SIZE"]])
    seq_len = tf.placeholder(tf.int32, [None])
    lstm = LSTM(data, config['LSTM'], seq_len=l)
    #
    prediction = lstm.forward

    #
    # loss = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    # optimizer = tf.train.AdamOptimizer()
    # minimize = optimizer.minimize(loss)
    #



    batch_size = 4
    total = len(train_input)
    epochs = 1
    init_op = tf.global_variables_initializer()
    #
    with tf.Session() as sess:
        sess.run(init_op)

        for e in range(epochs):
            for i in range(0, total, batch_size):
                inp,out = train_input[i:i+batch_size],train_output[i:i+batch_size]
                # loss_out = sess.run(loss, {data:inp, target:out})
                print sess.run(prediction,{data:inp})
                # prediction = sess.run(minimize,{data:inp, target:out})
    #


# main()
