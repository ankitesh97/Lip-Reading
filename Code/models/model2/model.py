

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "./loadData.py")


import tensorflow as tf
from CNN import CNN
from LSTM import LSTM
from util import *
from loadData import *
import time
import sys
from sklearn.metrics import confusion_matrix

CONFIG_FILE = 'config_prod.json'

dynamic_config = loadConfig(CONFIG_FILE)
train_flag = dynamic_config['Training']['is_training']
model_save_add = dynamic_config["Training"]['save_file_address']


class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            self.op = "cnn_forward/dense_2/BiasAdd:0"

    def run(self, data):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder
        # print self.graph.get_tensor_by_name('cnn_forward/conv2d/bias:0').eval(session=self.sess)

        output = []
        for i in range(data.shape[0]):
            output.append(self.sess.run(self.op, feed_dict={"Placeholder:0": data[i], "Placeholder_1:0":False}))
        return np.array(output)

def cnn_forward(input_array,sess2, op, input_te, is_train):
    # input_array dim : 29 X batchSize X width x height x channel
    output = []
    for i in range(input_array.shape[0]):
        # print "cool"
        output.append(sess2.run(op,feed_dict={input_te:input_array[i], is_train:False}))
    return np.array(output)



def dense_layer_op(input_te, is_train):

    config = loadConfig(CONFIG_FILE)
    dense_config = config["final_layer"]
    # with tf.variable_scope("final_layer_part"):
    dense_1_last =  tf.layers.dense(inputs=input_te, units= dense_config['units_layer_1'], activation=mapActivationFunc(dense_config['activation']),kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    dropout_1_last = tf.layers.dropout(inputs=dense_1_last, rate=dense_config['dropout_1_rate'], training=is_train)
    batch_norm_dense_1_last = tf.layers.batch_normalization(inputs=dropout_1_last, name="noreg_batch_norm_1")

    #dense layer 2
    dense_2_last =  tf.layers.dense(inputs=batch_norm_dense_1_last, units= dense_config['units_layer_2'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    dropout_2_last = tf.layers.dropout(inputs=dense_2_last, rate=dense_config['dropout_2_rate'], training=is_train)
    batch_norm_dense_2_last = tf.layers.batch_normalization(inputs=dropout_2_last, name="noreg_batch_norm_2")

    #dense layer 3

    dense_3_last =  tf.layers.dense(inputs=batch_norm_dense_2_last, units= dense_config['units_layer_3'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    batch_norm_dense_3_last = tf.layers.batch_normalization(inputs=dense_3_last, name="noreg_batch_norm_3")


    #dense layer 4
    dense_4_last =  tf.layers.dense(inputs=batch_norm_dense_3_last, units= dense_config['units_layer_4'], activation=mapActivationFunc(dense_config['activation']), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    #output layer
    features_last =  tf.layers.dense(inputs=dense_4_last, units= dense_config['feature_dim'], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return features_last

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

def main():

    config = loadConfig('config_prod.json')
    training_params = config["Training"]

    time_steps = config['LSTM']['time_steps']
    input_shape = config["CNN"]["Input_shape"]
    features = 24

    sequence = tf.placeholder(tf.float32, shape=[time_steps,None,512+features], name="Input_seq_hybrid")
    is_train = tf.placeholder(tf.bool, name="is_train_hybrid")
    seq_len = tf.placeholder(tf.int32, name="seq_len", shape=[None])
    last_layers = lambda input_tensor: dense_layer_op(input_tensor, is_train)
    last_layers_op = tf.map_fn(last_layers, sequence,  dtype=tf.float32, swap_memory=True)
    last_layers_output = tf.transpose(last_layers_op, [1, 0, 2])

    lstm_op_forward = LSTM(last_layers_output,config["LSTM"],seq_len=seq_len,is_training=is_train).forward

    nb_classes = training_params['nb_classes']
    learning_rate_init = tf.constant(training_params['learning_rate'], dtype=tf.float32)
    learning_rate = tf.Variable(training_params['learning_rate'],trainable=False, dtype=tf.float32)

    global_step = tf.Variable(0, trainable=False)
    global_step_increment = tf.assign(global_step, global_step+1)
    one = tf.constant(1,dtype=tf.float32)
    decay_const = tf.constant(1,dtype=tf.float32)
    # print one
    decay_op = tf.multiply(learning_rate_init, tf.divide(one,tf.add(one,tf.multiply(decay_const,tf.to_float(global_step)))))
    learning_rate_decay = tf.assign(learning_rate,decay_op)

    onehot_labels = tf.placeholder(tf.int32, [None, nb_classes], name="onehot")
    train_op, loss = train(onehot_labels, lstm_op_forward, learning_rate)

    #prediction
    probabilities = tf.nn.softmax(lstm_op_forward, name="predict_hybrid")
    classes = tf.argmax(probabilities, axis=1)


    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    batch_size = training_params['batch_size']
    acc_size = training_params['acc_size']
    # saver = tf.train.Saver(max_to_keep=4)
    init = tf.global_variables_initializer()
    Losses = []
    Losses_train = []
    model_save_add_local = './trained_models/cnn/clstmModel_final'

    cnn_model = ImportGraph(model_save_add_local)

    saver = tf.train.Saver(max_to_keep=8)
    max_acc = 0
    with tf.Session() as sess:
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
                #dimension of feature_vector #timexbatchx24
                X,y,sequence_length,feature_vector = getNextBatch(batch_size, is_colored)
                X = X/255.0
                X = cnn_model.run(X) # time X batch X 512
                X = np.concatenate((X,feature_vector),axis=-1)
                # X = makeAppZero(X,sequence_length)
                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={sequence:X,onehot_labels:one_hot_targets, is_train:True,seq_len:sequence_length})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()

            emptyDataQueue()
            total_val = loadDataQueue(is_val='val')
            X_acc,Y, sequence_length, feature_vector = getNextBatch(total_val,is_train='val')
            X_acc = X_acc/255.0
            X_acc = cnn_model.run(X_acc)
            X_acc = np.concatenate((X_acc,feature_vector),axis=-1)
            Y = Y.reshape(-1)
            one_hot_targets = np.eye(nb_classes)[Y]
            emptyDataQueue()
            loadDataQueue(is_val='train')
            X_acc_train,Y_train,sequence_length_train, feature_vector = getNextBatch(total_val)
            X_acc_train = X_acc_train/255.0

            X_acc_train = cnn_model.run(X_acc_train)
            X_acc_train = np.concatenate((X_acc_train,feature_vector),axis=-1)
            Y_train = Y_train.reshape(-1)
            one_hot_targets_train = np.eye(nb_classes)[Y_train]

            predictions = sess.run(probabilities, feed_dict={sequence:X_acc,is_train:False, seq_len:sequence_length})
            predicted_classes = sess.run(classes, feed_dict={sequence:X_acc, onehot_labels:one_hot_targets,is_train:False, seq_len:sequence_length})
            loss_val = sess.run(loss,feed_dict={sequence:X_acc, onehot_labels:one_hot_targets,is_train:False, seq_len:sequence_length})

            predictions_train = sess.run(probabilities, feed_dict={sequence:X_acc_train,is_train:False, seq_len:sequence_length_train})
            predicted_classes_train = sess.run(classes, feed_dict={sequence:X_acc_train, onehot_labels:one_hot_targets_train,is_train:False, seq_len:sequence_length_train})
            loss_val_train = sess.run(loss,feed_dict={sequence:X_acc_train, onehot_labels:one_hot_targets_train,is_train:False, seq_len:sequence_length_train})
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
    #
            sess.run(global_step_increment)
            sess.run(learning_rate_decay)

        saver.save(sess, model_save_add+"model_final")
        # print graph.get_tensor_by_name('rnn_forward/dense_1/bias:0').eval()

        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses), "losses_train":map(float,Losses_train)}))
        print max_acc


def makeAppZero(X,seq_len):
    X = np.swapaxes(X,0,1)
    for i in range(X.shape[0]):
        X[i][int(seq_len[i]):] = 0
    return X


def test():
    c = loadConfig(CONFIG_FILE)
    nb_classes = c["Training"]["nb_classes"]
    new_saver = tf.train.import_meta_graph(model_save_add+'model_final.meta')
    graph = tf.get_default_graph()

    model_save_add_local = './trained_models/cnn/clstmModel_final'

    cnn_model = ImportGraph(model_save_add_local)

    with tf.Session() as sess:


        new_saver.restore(sess, tf.train.latest_checkpoint(model_save_add))
        sequence = graph.get_tensor_by_name('Input_seq_hybrid:0')
        is_train = graph.get_tensor_by_name('is_train_hybrid:0')
        seq_len = graph.get_tensor_by_name('seq_len:0')

        onehot_labels = graph.get_tensor_by_name('onehot:0')

        total = loadDataQueue(is_val='val')
        X, y, sequence_length, feature_vector = getNextBatch(total,is_train='val')
        X = X/255.0
        X = cnn_model.run(X)
        X = np.concatenate((X,feature_vector),axis=-1)

        # X = np.concatenate((X,feature_vector),axis=-1)
        # X = makeAppZero(X,sequence_length)

        y2 = y.reshape(-1)
        one_hot_targets = np.eye(nb_classes)[y2]

        preds =  sess.run('predict_hybrid:0',feed_dict={sequence:X, is_train:False, onehot_labels:one_hot_targets,seq_len:sequence_length})
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
        print confusion_matrix(y,preds_classes)

if __name__ == '__main__':
    if train_flag:
        main()
    else:
        test()
