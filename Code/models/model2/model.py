

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


CONFIG_FILE = 'config_prod.json'

dynamic_config = loadConfig(CONFIG_FILE)
train_flag = dynamic_config['Training']['is_training']
model_save_add = dynamic_config["Training"]['save_file_address']

def cnn_forward(input_array,sess2, op, input_te, is_train):
    # input_array dim : 29 X batchSize X width x height x channel
    output = []
    for i in range(input_array.shape[0]):
        # print "cool"
        output.append(sess2.run(op,feed_dict={input_te:input_array[i], is_train:False}))
    return np.array(output)

def tryc():
    total = loadDataQueue(0)
    x,y = getNextBatch(10)
    print cnn_forward(x).shape

def dense_layer_op(input_te, is_train):

    config = loadConfig('config_prod.json')
    dense_config = config["final_layer"]
    # with tf.variable_scope("final_layer_part"):
    dense_1_last =  tf.layers.dense(inputs=input_te, units= dense_config['units_layer_1'], activation=mapActivationFunc(dense_config['activation']),kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    dropout_1_last = tf.layers.dropout(inputs=dense_1_last, rate=dense_config['dropout_1_rate'], training=is_train)
    batch_norm_dense_1_last = tf.contrib.layers.batch_norm(inputs=dropout_1_last)

    #dense layer 2
    dense_2_last =  tf.layers.dense(inputs=batch_norm_dense_1_last, units= dense_config['units_layer_2'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    dropout_2_last = tf.layers.dropout(inputs=dense_2_last, rate=dense_config['dropout_2_rate'], training=is_train)
    batch_norm_dense_2_last = tf.contrib.layers.batch_norm(inputs=dropout_2_last)

    #dense layer 3

    dense_3_last =  tf.layers.dense(inputs=batch_norm_dense_2_last, units= dense_config['units_layer_3'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    batch_norm_dense_3_last = tf.contrib.layers.batch_norm(inputs=dense_3_last)

    #output layer
    features_last =  tf.layers.dense(inputs=batch_norm_dense_3_last, units= dense_config['feature_dim'], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return features_last

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

    sequence = tf.placeholder(tf.float32, shape=[time_steps,None, 512], name="Input_seq_hybrid")
    is_train = tf.placeholder(tf.bool, name="is_train_hybrid")
    # cnn_op_
    last_layers = lambda input_tensor: dense_layer_op(input_tensor, is_train)
    last_layers_op = tf.map_fn(last_layers, sequence,  dtype=tf.float32, swap_memory=True)
    last_layers_output = tf.transpose(last_layers_op, [1, 0, 2])

    lstm_op_forward = LSTM(last_layers_output,config["LSTM"],is_training=is_train).forward

    nb_classes = training_params['nb_classes']
    learning_rate = training_params['learning_rate']

    onehot_labels = tf.placeholder(tf.int32, [None, nb_classes], name="onehot")
    train_op, loss = train(onehot_labels, lstm_op_forward, learning_rate)

    #prediction
    probabilities = tf.nn.softmax(lstm_op_forward, name="predict")
    classes = tf.argmax(probabilities, axis=1)


    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    batch_size = training_params['batch_size']
    acc_size = training_params['acc_size']
    # saver = tf.train.Saver(max_to_keep=4)
    init = tf.global_variables_initializer()
    Losses = []
    model_save_add_local = './trained_models/modelv1/'

    #load trained model
    # sess = tf.Session()
    # sess.close()

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_save_add_local+'clstmModel_final.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_save_add_local))
        graph = tf.get_default_graph()
        input_te_cnn = graph.get_tensor_by_name("Placeholder:0")
        is_train_cnn = graph.get_tensor_by_name("Placeholder_1:0")
        cnn_forward_op = graph.get_tensor_by_name('cnn_forward/cnn_final_layer:0')
        # print "start"
        # print graph.get_tensor_by_name('cnn_forward/conv2d/bias:0').eval()

        tf.stop_gradient(cnn_forward_op)
        sess.run(init)
        print "---------------Starting to train-------------"
        sys.stdout.flush()
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
                X = cnn_forward(X,sess,cnn_forward_op, input_te_cnn, is_train_cnn)
                # print "cnn done"
                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={sequence:X,onehot_labels:one_hot_targets, is_train:True})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()
            if(e%3==0):
                new_saver.save(sess, model_save_add+'clstmModel', global_step=e,write_meta_graph=False)
            # print graph.get_tensor_by_name('cnn_forward/conv2d/bias:0').eval()

            emptyDataQueue()
            loadDataQueue(is_colored)

            #take data
            X_acc,Y = getNextBatch(acc_size, is_colored)
            X_acc =X_acc/255.0
            X_acc = cnn_forward(X_acc,sess,cnn_forward_op, input_te_cnn, is_train_cnn)
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
    #
        new_saver.save(sess, model_save_add+"clstmModel_final")
        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses)}))




def test():

main()
# tryc()
