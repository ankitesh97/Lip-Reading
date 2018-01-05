
import tensorflow as tf
import numpy as np
from CNN import CNN
from util import *


def loadModel():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    config = loadConfig('config.json')
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('./my-model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        input_pl = graph.get_tensor_by_name("Placeholder:0")
        # input_tensor = tf.reshape(input_te, [-1,28,28,1])
        preds =  sess.run('prediction:0',feed_dict={input_pl:eval_data})
        classes = tf.argmax(preds, axis=1)
        preds_classes = sess.run(classes)
        count=0
        for i in range(len(preds_classes)):
            if preds_classes[i] == eval_labels[i]:
                count = count + 1

        print (count*1.0)/len(preds_classes)

loadModel()
