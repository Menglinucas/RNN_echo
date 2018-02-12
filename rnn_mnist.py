import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# load the data
mnist = input_data.read_data_sets('data',one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# input_x
x = tf.placeholder("float32", [None, n_input, n_steps])  # time direction: x
# outpuy y
y = tf.placeholder("float32", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# define LSTM, get output and state each timestep
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
_state = lstm_cell.zero_state(batch_size,tf.float32)
lstm_x = tf.transpose(x,[2,0,1])  # n_steps, batchsize, n_input
lstm_x = tf.reshape(lstm_x,[-1,n_input])
lstm_x = tf.matmul(lstm_x,weights['hidden']) + biases['hidden']
lstm_x = tf.split(lstm_x,n_steps,0)  # n_steps个列表(batchsize x n_input)
outputs, states = rnn.static_rnn(lstm_cell, lstm_x, initial_state = _state)  # output/state
# lstm_x need to satisfy a shape: n_step lists

predication = tf.matmul(outputs[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predication,labels=y))

#AdamOptimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
correct_pred = tf.equal(tf.argmax(predication,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
	    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	    # Reshape data to get 28 seq of 28 elements
	    batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
	    # Fit training using batch data
	    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
	    if step % display_step == 0:
	            # Calculate batch accuracy
	        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,})
	            # Calculate batch loss
	        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
	        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +  ", Training Accuracy= " + "{:.5f}".format(acc))
	    step += 1

	print("Optimization Finished!")

	#evaluate the accuracy
	test_len = batch_size
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(predication,1), tf.argmax(y,1))
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))