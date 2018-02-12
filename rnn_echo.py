import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000  # used for generate training data
truncated_backprop_length = 15  # time steps, t1, t2, t3 ... ... t15
state_size = 4  # equivalent to the number of hiden nodes
num_classes = 2
echo_step = 3  # used for generate training labels
batch_size = 5  # used for simultaneous training
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():  
	# 生成一个数组，元素为0,1，各占50%  
	x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))    
	# 沿着某一方向向前滚动echo_step  
	# 如：test=np.arange(0,5) np.roll(test,3) ---> array([2,3,4,0,1])
	y = np.roll(x, echo_step)  
	# 滚过的头部元素设为0  
	y[0:echo_step] = 0   

	x = x.reshape((batch_size, -1)) 
	y = y.reshape((batch_size, -1))
	print(x.shape)
	print(y.shape)
	return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])    
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])    
# 状态占位
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)    
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)    

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)    
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# unpacking
# Unpack columns    
inputs_series = tf.unstack(batchX_placeholder, axis=1)    
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass    
current_state = init_state    
states_series = []    
for current_input in inputs_series:    
	current_input = tf.reshape(current_input, [batch_size, 1])    
	input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns    
	next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition    
	states_series.append(next_state)  # 同时计算的状态值(batchsize)  
	current_state = next_state

# fully connected layer
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition    
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]    

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits) for logits, labels in zip(logits_series,labels_series)]    
total_loss = tf.reduce_mean(losses)    

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):    
	plt.subplot(2, 3, 1)    
	plt.cla()   # clear the current axes 
	plt.plot(loss_list)

	for batch_series_idx in range(5):    
		one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]    
		single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])    

		plt.subplot(2, 3, batch_series_idx + 2)    
		plt.cla()    
		plt.axis([0, truncated_backprop_length, 0, 2])    
		left_offset = range(truncated_backprop_length)    
		plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")    
		plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")    
		plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")    

		plt.draw()    
		plt.pause(0.0001)    

# run
with tf.Session() as sess:    
	sess.run(tf.global_variables_initializer())    
	plt.ion()    # thurn interactive mode on
	plt.figure()    # create a new figure
	plt.show()    # display a figure
	loss_list = []    

	for epoch_idx in range(num_epochs):    
		x,y = generateData()    
		_current_state = np.zeros((batch_size, state_size))    

		print("New data, epoch", epoch_idx)    

		for batch_idx in range(num_batches):   # series data to minibatch 
			start_idx = batch_idx * truncated_backprop_length    
			end_idx = start_idx + truncated_backprop_length    

			batchX = x[:,start_idx:end_idx]    
			batchY = y[:,start_idx:end_idx]    

			_total_loss, _train_step, _current_state, _predictions_series = sess.run(    
				[total_loss, train_step, current_state, predictions_series],    
				feed_dict={    
				batchX_placeholder:batchX,    
				batchY_placeholder:batchY,    
				init_state:_current_state    
				})    

			loss_list.append(_total_loss)    

			if batch_idx%100 == 0:    
				print("Step",batch_idx, "Loss", _total_loss)    
				plot(loss_list, _predictions_series, batchX, batchY)    

	plt.ioff()    
	plt.show()    