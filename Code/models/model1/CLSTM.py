
import tensorflow as tf
from CNN import CNN
from LSTM import LSTM
from util import *
from loadData import *


#defines the train operation
def train(prediction, target):
    loss = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(loss)
    return minimize, loss


def main():
    config = loadConfig('config.json')
    training_params = config["Training"]

    #define cnn operation
    inp_shape = config['CNN']['Input_shape']
    time_steps = config['LSTM']['time_steps']

    sequence = tf.placeholder(tf.float32, shape=[time_steps, None]+inp_shape)
    target = tf.placeholder(tf.float32, shape=[None, config["LSTM"]["vocab_size"]])

    cnn_forward_op = lambda input_tensor: CNN(input_tensor,config["CNN"]).forward
    cnn_output = tf.map_fn(cnn_forward_op, sequence, dtype=tf.float32, swap_memory=True)
    lstm_op_forward = LSTM(config["LSTM"], cnn_output)

    probabilities = tf.nn.softmax(lstm_op_forward, name="predict")
    train_op, loss = train(probabilities, target)

    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    batch_size = training_params['batch_size']

    saver = tf.train.Saver()
    with tf.Session() as sess:

        for e in range(epochs):
            total = loadDataQueue(is_colored)

            for i in range(0, total, batch_size):
                if()
                X,y = getNextBatch(batch_size, is_colored)
                sess.run(train_op, feed_dict={sequence:X, target:y})

    #define rnn operation
