
import tensorflow as tf
import json
from matplotlib import pyplot

def mapActivationFunc(activation_name):

	return {
		'relu': tf.nn.relu,
		'relu6': tf.nn.relu6,
		'crelu': tf.nn.crelu,
		'elu': tf.nn.elu,
		'softplus': tf.nn.softplus,
		'softsign': tf.nn.softsign,
		'dropout': tf.nn.dropout,
		'bias_add': tf.nn.bias_add,
		'sigmoid': tf.nn.sigmoid,
		'tanh': tf.nn.tanh
	}.get(activation_name,'not found')


def loadConfig(filename):
    with open(filename) as json_data:
        d = json.load(json_data)
        x = range(0, len(d["losses"]))
        y = d["losses"]
        pyplot.plot(x,y)
        pyplot.show()

loadConfig('losses.txt')
