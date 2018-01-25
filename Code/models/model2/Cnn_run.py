


def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "./loadDataLip.py")

import tensorflow as tf
from CNN import CNN
from loadData import *
from util import *
import sys

CONFIG_FILE = 'config_prod.json'

dynamic_config = loadConfig(CONFIG_FILE)
train_flag = dynamic_config['Training']['is_training']
model_save_add = dynamic_config["Training"]['save_file_address']


def main():
    config = loadConfig(CONFIG_FILE)
    inp_shape = config['CNN']['Input_shape']
    learning_rate = config['Training']['learning_rate']
    nb_classes = 2
    input_te = tf.placeholder(tf.float32, shape=[None,]+inp_shape)
    is_train = tf.placeholder(tf.bool)
    cnn_object = CNN(input_te,config["CNN"], is_train)
    forward_op = cnn_object.forward
    onehot_labels = tf.placeholder(tf.int32, [None,nb_classes])
    predict = tf.nn.softmax(forward_op)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=forward_op)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step()
    )

    #prediction
    classes = tf.argmax(predict, axis=1)

    n_epochs = config["Training"]["epochs"]
    batch_size = config["Training"]["batch_size"]
    acc_size = config["Training"]["acc_size"]
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
    Losses = []
    with tf.Session() as sess:
        print "---------------Starting to train-------------"
        sys.stdout.flush()
        sess.run(init)
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

        #epoch loop
        for e in range(n_epochs):
            total = loadDataQueue()
            if e%3 == 0:
                with open("control.txt",'r') as f:
                    control = f.read()
                    if control.strip() == "1":
                       print "-----------------stopping the training process .........."
                       sys.stdout.flush()
                       break

            batch_count = 0
            for i in range(0, total, batch_size):
                X,y = getNextBatch(batch_size) #batchSize X width X height
                X = X/255.0
                y = y.reshape(-1)
                one_hot_targets = np.eye(nb_classes)[y]
                sess.run(train_op, feed_dict={input_te:X,onehot_labels:one_hot_targets, is_train:True})
                batch_count = batch_count + 1
                if(batch_count%2==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()
            if(e%3==0):
                saver.save(sess, model_save_add+'clstmModel', global_step=e,write_meta_graph=False)

            emptyDataQueue()
            loadDataQueue()

            #take data
            X_acc,Y = getNextBatch(acc_size)
            X_acc =X_acc/255.0
            Y = Y.reshape(-1)
            one_hot_targets = np.eye(nb_classes)[Y]

            # predictions = sess.run(probabilities, feed_dict={input_te:X_acc,is_train:False})
            predicted_classes = sess.run(classes, feed_dict={input_te:X_acc, onehot_labels:one_hot_targets,is_train:False})
            loss_val = sess.run(loss,feed_dict={input_te:X_acc, onehot_labels:one_hot_targets,is_train:False})
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
        open("losses.txt", "w").write(json.dumps({"losses":map(float,Losses)}))

def test():


    c = loadConfig(CONFIG_FILE)
    color_flag = c['Training']['is_colored']
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph(model_save_add+'clstmModel_final.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_save_add))
        graph = tf.get_default_graph()
        input_te = graph.get_tensor_by_name("Placeholder:0")
        is_train = graph.get_tensor_by_name("Placeholder_1:0")
        onehot_labels = graph.get_tensor_by_name("Placeholder_2:0")
        total = loadDataQueue()
        X, y = getNextBatch(total)

        X = X/255.0
        y2 = y.reshape(-1)
        one_hot_targets = np.eye(2)[y2]
        # input_tensor = tf.reshape(input_te, [-1,28,28,1])
        preds =  sess.run('Softmax:0',feed_dict={input_te:X, is_train:False, onehot_labels:one_hot_targets})
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
