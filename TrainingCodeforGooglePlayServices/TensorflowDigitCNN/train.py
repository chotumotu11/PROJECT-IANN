#Author : Dipayan Deb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Model Input and output.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Model parameters.
W = tf.Variable(tf.zeros([784,10]),name="W")
b = tf.Variable(tf.zeros([10]),name="b")
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for _ in range(1000):
  batch = mnist.train.next_batch(100)
  #print("Now I will print the data for better understanding of me")
  print(type(batch[0][0][0]))
  #print(batch[0])
  sess.run(train_step,{x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Digit recognition accuracy in percentage :",sess.run(accuracy,{x: mnist.test.images, y_: mnist.test.labels})*100)

#Now saving the variable.
saver = tf.train.Saver()
save_path = saver.save(sess, "./temp/model.ckpt")
print("Model saved in file: %s" % save_path)



