import tensorflow as tf
import numpy as np
from wordDict import *
INPUT_DIMENSION = 100 #This defines the number of nodes in the final layer of CNN
SEQUENCE_LENGTH = 29 #The length of one sequence of data in our case 29 frames of the video
VOCABULARY_SIZE = 10
HIDDEN_DIMENSTION = 200
NO_EXAMPLES = 1000
TRAIN_EXAMPLES=500
BATCH_SIZE = 500
#Static data

train_input = np.random.rand(NO_EXAMPLES,SEQUENCE_LENGTH,INPUT_DIMENSION)
randomNumbers = np.random.randint(VOCABULARY_SIZE,size=NO_EXAMPLES)
reshapedRandomNumbers = randomNumbers.reshape(-1)
train_output = np.eye(VOCABULARY_SIZE)[reshapedRandomNumbers]

test_input = train_input[TRAIN_EXAMPLES:]
test_output = train_output[TRAIN_EXAMPLES:]

train_input = train_input[:TRAIN_EXAMPLES]
train_output = train_output[:TRAIN_EXAMPLES]
#Building the model graph

data = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH,INPUT_DIMENSION])
target = tf.placeholder(tf.float32,[None,VOCABULARY_SIZE])
cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIMENSTION,state_is_tuple=True)
output,state = tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1) #Gather takes two values param and indices. works like slicing
weight = tf.Variable(tf.truncated_normal([HIDDEN_DIMENSTION,int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1,shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target,1),tf.argmax(prediction,1))
error = tf.reduce_mean(tf.cast(mistakes,tf.float32))

#executing the model graph

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
batch_size = BATCH_SIZE
no_of_batches = int(len(train_input)/batch_size)
epoch = 10
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp,out = train_input[ptr:ptr+batch_size],train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data:inp ,target:out})
    print "Epoch -",str(i)
incorrect = sess.run(error,{data:test_input, target:test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
