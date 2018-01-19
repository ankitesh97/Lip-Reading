

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
load_src("loadData", "../../utils/loadData.py")

import tensorflow as tf
import os
from CNN import CNN
from loadData import *
from util import *
import json

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
        out = []
        for i in range(data.shape[0]):
            out.append(self.sess.run(self.op, feed_dict={"Placeholder:0": data[0], "Placeholder_1:0":False}))

        return np.swapaxes(np.array(out),0,1)

def main():
    config = loadConfig('config_prod.json')
    total = loadDataQueue()
    print "====================="
    print total
    print "====================="

    batch_size = 256
    cnn = ImportGraph('./trained_models/cnn_model_lip_border/clstmModel_final')
    to = 0
    for bat in range(0,total, batch_size):
        X,_,seq_len = getNextBatch(batch_size)
        X = X/255.0
        features_out = cnn.run(X) #batchSize X time step X 512
        # print features_out.shape
        for i in range(X.shape[1]):
            seq_len_curr = seq_len[i]
            for j in range(seq_len_curr):
                to = to + 1
                f = open("features_lip_border_all_words/"+"feature_"+str(to)+".txt",'w')
                f.write(json.dumps({"feature":map(float,features_out[i][j])}))
                f.close()





main()
