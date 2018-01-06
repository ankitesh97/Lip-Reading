

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "../../utils/loadData.py")


import tensorflow as tf
from CNN import CNN
from LSTM import LSTM
from util import *
from loadData import *

#defines the train operation
def train(onehot_labels, predicted, learning_rate):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=predicted)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    minimize = optimizer.minimize(loss)
    return minimize, loss


def main():
    config = loadConfig('config.json')
    training_params = config["Training"]

    #define cnn operation
    inp_shape = config['CNN']['Input_shape']
    time_steps = config['LSTM']['time_steps']

    sequence = tf.placeholder(tf.float32, shape=[time_steps, None]+inp_shape, name="input_seq")
    target = tf.placeholder(tf.float32, shape=[None, config["LSTM"]["vocab_size"]], name="target")
    is_train = tf.placeholder(tf.bool)
    cnn_forward_op = lambda input_tensor: CNN(input_tensor,config["CNN"],is_training=is_train).forward
    cnn_output = tf.map_fn(cnn_forward_op, sequence, dtype=tf.float32, swap_memory=True)
    cnn_output = tf.transpose(cnn_output, [1, 0, 2])

    lstm_op_forward = LSTM(cnn_output,config["LSTM"],is_training=is_train).forward

    nb_classes = training_params['nb_classes']
    learning_rate = training_params['learning_rate']

    onehot_labels = tf.placeholder(tf.int32, [None, nb_classes], name="onehot")

    #prediction
    probabilities = tf.nn.softmax(lstm_op_forward, name="predict")
    classes = tf.argmax(probabilities, axis=1)

    train_op, loss = train(onehot_labels, lstm_op_forward, learning_rate)

    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    batch_size = training_params['batch_size']
    acc_size = training_params['acc_size']

    saver = tf.train.Saver(max_to_keep=4)
    init = tf.global_variables_initializer()
    Losses = []
    with tf.Session() as sess:

        print "---------------Starting to train-------------"
        sess.run(init)
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        for e in range(epochs):
            total = loadDataQueue(is_colored)
            if(e%3==0):
                with open("control.txt",'r') as f:
                    control = f.read()
                    if control.strip() == "1":
                       print "-----------------stopping the training process .........."
                       break
            batch_count = 0
            for i in range(0, total, batch_size):
                X,y = getNextBatch(batch_size, is_colored)
                X = X/255.0
                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={sequence:X,onehot_labels:one_hot_targets, is_train:True})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
            saver.save(sess, './trained_models/clstmModel', global_step=e,write_meta_graph=False)

            emptyDataQueue()
            loadDataQueue(is_colored)

            #take data
            X_acc,Y = getNextBatch(acc_size, is_colored)
            X_acc =X_acc/255.0
            Y = Y.reshape(-1)
            one_hot_targets = np.eye(nb_classes)[Y]

            predictions = sess.run(probabilities, feed_dict={sequence:X_acc,is_train:False})
            predicted_classes = sess.run(classes, feed_dict={sequence:X_acc, onehot_labels:one_hot_targets,is_train:False})
            loss_val = sess.run(loss,feed_dict={sequence:X_acc, onehot_labels:one_hot_targets,is_train:False})
            Losses.append(loss_val)
            count=0
            for i in range(len(Y)):
                if predicted_classes[i] == Y[i]:
                    count = count + 1

            acc_train = (count*(1.0)/len(predicted_classes)) *100
            emptyDataQueue()

            print "Epoch: "+str(e)+" Loss: "+str(loss_val)+" Train Accuracy: "+str(acc_train)+"%"

        saver.save(sess, "./trained_models/clstmModel_final")
        print Losses
        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses)}))

    #define rnn operation


main()
