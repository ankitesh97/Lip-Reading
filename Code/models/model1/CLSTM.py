
import tensorflow as tf
from CNN import CNN
from LSTM import LSTM
from util import *
from loadData import *


#defines the train operation
def train(onehot_labels, predicted, learning_rate):
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=predicted)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

    nb_classes = training_params['nb_classes']
    learning_rate = training_params['learning_rate']

    onehot_labels = tf.placeholder(tf.int32, [None, nb_classes])

    #prediction
    probabilities = tf.nn.softmax(lstm_op_forward, name="predict")
    classes = tf.argmax(probabilities, axis=1)

    train_op, loss = train(onehot_labels, lstm_op_forward, learning_rate)

    epochs = training_params['epochs']
    is_colored = training_params['is_colored']
    batch_size = training_params['batch_size']
    acc_size = training_params['acc_size']

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        for e in range(epochs):
            total = loadDataQueue(is_colored)
            if(e%5==0):
                with open("controlTraining.txt",'r') as f:
                control = f.read()
                if control.strip() == "1":
                   print "-----------------stopping the training process .........."
                   break
            for i in range(0, total, batch_size):
                X,y = getNextBatch(batch_size, is_colored)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={sequence:X, target:y})

            loadDataQueue(is_colored)

            #take data
            X_acc,Y = getNextBatch(acc_size, is_colored)
            one_hot_targets = np.eye(nb_classes)[Y]

            predictions = sess.run(probabilities, feed_dict={sequence:X_acc})
            predicted_classes = sess.run(classes, feed_dict={sequence:X_acc, onehot_labels:one_hot_targets})
            loss_val = sess.run(loss, accuracy,feed_dict={sequence:X_acc, onehot_labels:one_hot_targets})

            count=0
            for i in range(len(Y)):
                if preds_classes[i] == Y[i]:
                    count = count + 1

            acc_train = count*(1.0)/len(preds_classes)

            print "Epoch: "+str(e)+" Loss: "+str(loss_val)+" Train Accuracy: "+str(acc_train)+"%"


    #define rnn operation
