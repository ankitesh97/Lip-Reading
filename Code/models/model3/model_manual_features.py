

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "./loadFeaturesBatch.py")


import tensorflow as tf
from CNN import CNN
from LSTM import LSTM
from Clustering import *
from util import *
from loadData import *
import time
import sys


CONFIG_FILE = 'config_prod.json'

dynamic_config = loadConfig(CONFIG_FILE)
train_flag = dynamic_config['Training']['is_training']
model_save_add = dynamic_config["Training"]['save_file_address']


def get_viseme_class(Kmeansobj, data, seq_len,y):
    output = []
    #data of size time X batch x feature
    print y
    for i in range(data.shape[0]):
        output.append(Kmeansobj.predict(data[i]))
    #shape of output time X batch X n_clusters
    output = np.array(output)
    output = np.swapaxes(output, 0, 1) #batch X time X n_clusters

    #make zero after defined time step
    for i in range(data.shape[1]):
        output[i][int(seq_len[i]):] = 0
    return output


def train(onehot_labels, predicted, learning_rate):
    # epsilon = tf.constant(value=0.00001)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=predicted))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimize = optimizer.minimize(loss)
    return minimize, loss



def main():

    max_time = 29
    feature_dim = 24
    config = loadConfig('config_prod.json')
    training_params = config["Training"]

    time_steps = config['LSTM']['time_steps']
    input_shape = config["CNN"]["Input_shape"]
    no_of_clusters = config['Clustering']["no_of_clusters"]
    clustering_iterations = config['Clustering']['clustering_iterations']
    clustering_save_file_add = config['Clustering']['clustering_save_file_add']
    clustering_obj = KMEANS(clustering_save_file_add,no_of_clusters)
    clustering_obj.loadFile()
    batch_size = training_params['batch_size']


    sequence = tf.placeholder(dtype=tf.float32, shape=[None,max_time,no_of_clusters], name="sequence")
    is_train = tf.placeholder(tf.bool, name="is_train")
    seq_len = tf.placeholder(tf.int32, name="seq_len", shape=[None])


    lstm_op_forward = LSTM(sequence,config["LSTM"], seq_len=seq_len, is_training=is_train).forward

    nb_classes = training_params['nb_classes']
    # learning_rate = training_params['learning_rate']
    learning_rate_init = tf.constant(training_params['learning_rate'], dtype=tf.float32)
    learning_rate = tf.Variable(training_params['learning_rate'],trainable=False, dtype=tf.float32)

    global_step = tf.Variable(0, trainable=False)
    global_step_increment = tf.assign(global_step, global_step+1)
    one = tf.constant(1,dtype=tf.float32)
    decay_const = tf.constant(1,dtype=tf.float32)
    decay_op = tf.multiply(learning_rate_init, tf.divide(one,tf.add(one,tf.multiply(decay_const,tf.to_float(global_step)))))
    learning_rate_decay = tf.assign(learning_rate,decay_op)

    onehot_labels = tf.placeholder(tf.int32, [None, nb_classes], name="onehot")

    #prediction
    probabilities = tf.nn.softmax(lstm_op_forward, name="predict_classes")
    train_op, loss = train(onehot_labels, lstm_op_forward, learning_rate)
    classes = tf.argmax(probabilities, axis=1)

    init = tf.global_variables_initializer()
    Losses = []
    saver = tf.train.Saver(max_to_keep=8)
    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    acc_size = training_params['acc_size']

    with tf.Session() as sess:
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
                X = get_viseme_class(clustering_obj,X,sequence_length,y) #returns batch x time x n_clsuters

                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                if len(np.argwhere(np.isnan(X))):
                    print "mila"
                sess.run(train_op, feed_dict={sequence:X,onehot_labels:one_hot_targets, is_train:True, seq_len:sequence_length})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()
            if(e%3==0):
                saver.save(sess, model_save_add+'model', global_step=e,write_meta_graph=False)
            emptyDataQueue()
            loadDataQueue(data='val')

            X_acc,Y, sequence_length = getNextBatch(acc_size)
            X_acc = get_viseme_class(clustering_obj,X_acc,sequence_length,Y) #returns batch x time x n_clsuters

            Y = Y.reshape(-1)
            one_hot_targets = np.eye(nb_classes)[Y]

            predictions = sess.run(probabilities, feed_dict={sequence:X_acc,is_train:False, seq_len:sequence_length})
            predicted_classes = sess.run(classes, feed_dict={sequence:X_acc, onehot_labels:one_hot_targets,is_train:False, seq_len:sequence_length})
            loss_val = sess.run(loss,feed_dict={sequence:X_acc, onehot_labels:one_hot_targets,is_train:False, seq_len:sequence_length})
            Losses.append(loss_val)
            count=0
            for i in range(len(Y)):
                if predicted_classes[i] == Y[i]:
                    count = count + 1

            acc_train = (count*(1.0)/len(predicted_classes)) *100
            emptyDataQueue()

            print "Epoch: "+str(e)+" Loss: "+str(loss_val)+" Train Accuracy: "+str(acc_train)+"%"+" Count "+str(count)
            sys.stdout.flush()
    #
            # sess.run(global_step_increment)
            # sess.run(learning_rate_decay)

        saver.save(sess, model_save_add+"model_final")
        # print graph.get_tensor_by_name('rnn_forward/dense_1/bias:0').eval()
        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses)}))




def test():
    c = loadConfig(CONFIG_FILE)
    config = c
    nb_classes = c["Training"]["nb_classes"]
    new_saver = tf.train.import_meta_graph(model_save_add+'model_final.meta')
    graph = tf.get_default_graph()


    training_params = config["Training"]

    time_steps = config['LSTM']['time_steps']
    input_shape = config["CNN"]["Input_shape"]
    no_of_clusters = config['Clustering']["no_of_clusters"]
    clustering_iterations = config['Clustering']['clustering_iterations']
    clustering_save_file_add = config['Clustering']['clustering_save_file_add']
    clustering_obj = KMEANS(clustering_save_file_add,no_of_clusters)
    clustering_obj.loadFile()
    batch_size = training_params['batch_size']

    with tf.Session() as sess:


        new_saver.restore(sess, tf.train.latest_checkpoint(model_save_add))
        sequence = graph.get_tensor_by_name('sequence:0')
        is_train = graph.get_tensor_by_name('is_train:0')
        seq_len = graph.get_tensor_by_name('seq_len:0')
        onehot_labels = graph.get_tensor_by_name('onehot:0')

        total = loadDataQueue(data='val')
        X, y, sequence_length = getNextBatch(total)
        X = get_viseme_class(clustering_obj,X,sequence_length,y)
        y2 = y.reshape(-1)
        one_hot_targets = np.eye(nb_classes)[y2]

        preds =  sess.run('predict_classes:0',feed_dict={sequence:X, is_train:False, onehot_labels:one_hot_targets, seq_len:sequence_length})
        classes = tf.argmax(preds, axis=1)
        preds_classes = sess.run(classes)
        print len(preds_classes)
        print preds_classes
        print "actual"
        print len(y)
        print y
        count=0
        for i in range(len(preds_classes)):
            if preds_classes[i] == y[i]:
                count = count + 1
        print count
        print ((count*1.0)/len(preds_classes)) * 100

if __name__ == '__main__':
    if train_flag:
        main()
    else:
        test()
