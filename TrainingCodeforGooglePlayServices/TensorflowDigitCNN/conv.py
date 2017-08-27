# Author : Dipayan Deb

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Decalring the weights.
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	intial = tf.constant(0.1,shape=shape)
	return tf.Variable(intial)
# Convolutions funciton with stride =1  and padding does not decrease spatial dimnsions.

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
# Crreating the fucntion for pooling.

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# The first cov layer. Each filter is of 5*5*1 . There are 32 filters in the first layer.

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function
# and finally max pool. The max_pool_2x2 method will reduce the image size to 14x14

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# The Second Convolutional Layer.

# Has 64 filters

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densly connected layer 

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#Dropout 

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer 

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop , W_fc2)+ b_fc2

# Trainig and running the model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  plt.imshow(np.reshape(batch[0][1],(28,28)))
  print(np.reshape(batch[0][1],(28,28)).shape)
  plt.show()
  if i%100 == 0:
  	 train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
  	 print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))