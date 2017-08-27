# Author : Dipayan Deb
# Purpose : Takes input an 28*28 image and outputs the digit if any in that image.

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
#print("Number of argumnets: ", len(sys.argv),"arguments")
import numpy as np
#print("Argument list : ",str(sys.argv))
print("The image name you passed : ",sys.argv[1])
#tensorflow accepts input as a queue. It accepts names of image inputs as a list.
filename_queue = tf.train.string_input_producer([sys.argv[1]])
#WholeFileReader is the type of tensorflow reader we use to read the whole file. There are others readers for CSV and othgetr formats
reader = tf.WholeFileReader()
#The reader returns two types or values. The second value is the actual binary image encoded in jpeg format.
# WE have to encode itself.
key,value = reader.read(filename_queue)
# WE encode the binary image to a tensor of format [height,width,channels].
img = tf.image.decode_jpeg(value,channels=3)
with tf.Session() as sess:
    #Starts a thread to read all the filenames from the queue in a non blocking manner.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # The above computation graph gets run.
    for i in range(1):
         image = sess.run(img)
   #We convert the tensor into a array so that we can feed it as input to our neural network.
    nparray = np.asarray(image)
    my_array = nparray

    # We eliminate the 3 channels and keep only one Therefore ultimately converting it into greyscale.
    main_image = []
    for i in range(28):
        new = []
        for j in range(28):
            new.append(my_array[i][j][0])
        main_image.append(new)
    #The above code converts the image into a list. So we reconvert it into a array.
    coord.request_stop()
    main_image_array = np.asarray(main_image)
    coord.join(threads)

# This is the input for tensorflow.
x = tf.placeholder(tf.float32,[None,784])

# We convert the input into a 1D array(vector).
input_data1 = tf.reshape(main_image_array,[1,784])
# We typecast the values of the array into float32 from uint8.
input_data = tf.cast(input_data1,tf.float32)



# Declare the variables. So that we can restore them.
W = tf.Variable(tf.zeros([784,10]),name="W")

b = tf.Variable(tf.zeros([10]),name="b")

# The neural network calculation.
y = tf.matmul(x,W)+ b
# The application of softmax fucntion. Does the same work as sigmoid fucntion but much better.
y1 = tf.nn.softmax(y)
#We now restore the variables from the training data.
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./temp/model.ckpt")
print("Model restored")

#We now run the computation graph to evaluate input_data. main_data_input will hold the 1d vector with float32 values.
for i in range(1):
    main_data_input = (255-sess.run(input_data))/255

# We now run the actual neural network.
for i in range(1):
    hello = sess.run(y1,{x: main_data_input})

# The index of the output y1 maps to the max value in y1.
buga = tf.argmax(hello,1)

# run the graph as defined for buga.
for i in range(1):
    print("The predicted value is : ",sess.run(buga))
