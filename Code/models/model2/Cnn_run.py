


def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "../../utils/loadData.py")

import tensorflow as tf
from CNN import CNN


CONFIG_FILE = 'config.json'

dynamic_config = loadConfig(CONFIG_FILE)
train_flag = dynamic_config['Training']['is_training']
model_save_add = dynamic_config["Training"]['save_file_address']


def main():
    config = loadConfig('config.json')
    inp_shape = config['CNN']['Input_shape']
    learning_rate = config['Training']['learning_rate']
    nb_classes = config['Training']['nb_classes']
    input_te = tf.placeholder(tf.float32, shape=[None,]+inp_shape)
    is_train = tf.placeholder(tf.bool)
    cnn_object = CNN(input_tensor,config["CNN"], is_train)
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

    n_epochs = config["Training"]["n_epochs"]
    batch_size = config["Training"]["batch_size"]
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
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
                if(batch_count%100==0):
                    print "Epoch: "+str(e)+" Batch: "+str(batch_count)
                    sys.stdout.flush()
            if(e%3==0):
                saver.save(sess, model_save_add+'clstmModel', global_step=e,write_meta_graph=False)

            emptyDataQueue()
            loadDataQueue()
