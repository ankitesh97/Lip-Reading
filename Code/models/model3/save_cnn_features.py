

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("util", "../../utils/model1.py")
    load_src("loadData", "../../utils/loadDataLip.py")

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
            self.op = "cnn_forward/cnn_final_layer:0"

    def run(self, data):
        return self.sess.run(self.op, feed_dict={"Placeholder:0": data, "Placeholder_1:0":False})


def main():
    config = loadConfig('config_prod.json')
    total = loadDataQueue()
    print total
    batch_size = 1
    cnn = ImportGraph('./trained_models/cnnv2/clstmModel_final')
    for i in range(0,total, batch_size):
        X,_ = getNextBatch(batch_size)
        X = X/255.0
        features = cnn.run(X)[0]
        f = open("features/"+"feature_"+str(i)+".txt",'w')
        f.write(json.dumps({"feature":map(float,features),"dim":len(features)}))
        f.close()


main()
