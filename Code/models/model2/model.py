

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
import time
import sys


def cnn_forward(input_te):

    model_save_add_local = './trained_models/modelv1/'
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph(model_save_add_local+'clstmModel_final.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_save_add_local))
        graph = tf.get_default_graph()
        input_te = graph.get_tensor_by_name("Placeholder:0")
        output =  sess.run('cnn_forward/cnn_final_layer:0',feed_dict={input_te:X, is_train:False})
        return output

def dense_layer_op(input_te, is_train):

    config = loadConfig('config_prod.json')
    dense_config = config["final_layer"]
    with tf.variable_scope("final_layer_part"):
        dense_1 =  tf.layers.dense(inputs=input_te, units= dense_config['units_layer_1'], activation=mapActivationFunc(dense_config['activation']),kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        dropout_1 = tf.layers.dropout(inputs=dense_1, rate=dense_config['dropout_1_rate'], training=is_train)
        batch_norm_dense_1 = tf.contrib.layers.batch_norm(inputs=dropout_1)

        #dense layer 2
        dense_2 =  tf.layers.dense(inputs=batch_norm_dense_1, units= dense_config['units_layer_2'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        dropout_2 = tf.layers.dropout(inputs=dense_2, rate=dense_config['dropout_2_rate'], training=is_train)
        batch_norm_dense_2 = tf.contrib.layers.batch_norm(inputs=dropout_2)

        #dense layer 3

        dense_3 =  tf.layers.dense(inputs=batch_norm_dense_1, units= dense_config['units_layer_3'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        batch_norm_dense_3 = tf.contrib.layers.batch_norm(inputs=dense_3)

        #output layer
        features =  tf.layers.dense(inputs=batch_norm_dense_3, units= dense_config['feature_dim'], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

        return features

def train(onehot_labels, predicted, learning_rate):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=predicted)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    minimize = optimizer.minimize(loss)
    return minimize, loss



def main():

    config = loadConfig('config_prod.json')
    training_params = config["Training"]

    time_steps = config['LSTM']['time_steps']
    input_shape = config["CNN"]["Input_shape"]

    sequence = tf.placeholder(tf.float32, shape=[None, time_steps]+input_shape)

    cnn_op = lambda input_tensor: cnn_forward(input_tensor)
    last_layers = lambda input_tensor: dense_layer_op(input_tensor)
    last_layers_output = tf.transpose(last_layers, [1, 0, 2])
    lstm_op_forward = LSTM(last_layers_output,config["LSTM"],is_training=is_train).forward

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
        sys.stdout.flush()
        sess.run(init)
        writer = tf.summary.FileWriter('logs/model2', graph=tf.get_default_graph())
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
            for i in range(0, total, batch_size):
                X,y = getNextBatch(batch_size, is_colored)
                X = X/255.0
                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={sequence:X,onehot_labels:one_hot_targets, is_train:True})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()
            if(e%3==0):
                saver.save(sess, model_save_add+'clstmModel', global_step=e,write_meta_graph=False)

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

            print "Epoch: "+str(e)+" Loss: "+str(loss_val)+" Train Accuracy: "+str(acc_train)+"%"+" Count "+str(count)
            sys.stdout.flush()

        saver.save(sess, model_save_add+"clstmModel_final")
        print Losses
        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses)}))

    #define rnn operation
