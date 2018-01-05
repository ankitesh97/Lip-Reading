import tensorflow as tf

def LossFunction(loss_name):
	return {
		'12_loss': tf.nn.12_loss,
		'log_poisson_loss': tf.nn.log_poisson_loss
	}.get(loss_name, 'invalid loss function')
