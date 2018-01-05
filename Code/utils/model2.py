import tensorflow as tf

def ActivationFunction(activation_name):
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
	
#print ActivationFunction('asdsd')
