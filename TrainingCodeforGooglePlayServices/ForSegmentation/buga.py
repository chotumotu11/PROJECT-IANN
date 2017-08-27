# Author : Dipayan Deb
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from os.path import expanduser


MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      print('WARNING: Folder {} has more than {} images. Some images will '
            'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


path = expanduser("~")+"/tf_files/flower_photos"

a = create_image_lists(path,20,20)

size=len(a.keys())
d= []
i=0
location_array = [] #Contans the exact location of the folder.
for k in a.keys():
    d.append(k)
    location_array.append(path+"/"+k)
    i=i+1

def create_path(l_array,a,type):
    main_return  = []
    main_result_class = []
    j = 0
    while j < size:
        main_return.append([])
        main_result_class.append([])
        for k in a[d[j]][type]:
            result =l_array[j] + "/" + k
            main_return[j].append(result)
            main_result_class[j].append(j)
        j = j + 1

    return main_return,main_result_class

#Create path for Training dataset.
training_path = create_path(location_array,a,'training')
# Create path for testing dataset
testing_path=create_path(location_array,a,'testing')
# Create path for validation
validation_path=create_path(location_array,a,'validation')



# Now loading the images in tensorflow :)
def create_image_queue(path):
    size1 = len(path[0])
    image_queue = []
    for i in range(0, size1):
        for j in range(0, len(path[0][i])):
            image_queue.append(path[0][i][j])
    return image_queue

training_image_queue = create_image_queue(training_path)
testing_image_queue = create_image_queue(testing_path)
validation_image_queue = create_image_queue(validation_path)

def create_image_class_queue(path):
    size1 = len(path[1])
    class_queue = []
    for i in range(0, size1):
        for j in range(0, len(path[1][i])):
            class_queue.append(path[1][i][j])
    return class_queue

training_image_class_queue = create_image_class_queue(training_path)
testing_image_class_queue = create_image_class_queue(testing_path)
validation_image_class_queue = create_image_class_queue(validation_path)


def get_image_as_2d_array(training_image_queue):
    # tensorflow accepts input as a queue. It accepts names of image inputs as a list.
    filename_queue_training = tf.train.string_input_producer(training_image_queue,shuffle=False)

    # WholeFileReader is the type of tensorflow reader we use to read the whole file. There are others readers for CSV and othgetr formats
    reader = tf.WholeFileReader()
    # The reader returns two types or values. The second value is the actual binary image encoded in jpeg format.
    # WE have to encode itself.
    key, value = reader.read(filename_queue_training)
    # WE encode the binary image to a tensor of format [height,width,channels].
    img1 = tf.image.decode_jpeg(value, channels=3)
    img = tf.image.resize_images(img1,(28,28))
   # img = tf.image.resize_image_with_crop_or_pad(img1,28,28)
    # A 2d array to store all the 1D images.
    d1_images = []
    with tf.Session() as sess:
        # Starts a thread to read all the filenames from the queue in a non blocking manner.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # The above computation graph gets run.
        msize = len(training_image_queue)
        print(msize)
        for i in range(msize):
            image = sess.run(img)
            # We convert the tensor into a array so that we can feed it as input to our neural network.
            my_array = np.asarray(image)
            main_image=rgb2gray(my_array)
            # The above code converts the image into a list. So we reconvert it into a array.
            main_image_array = np.subtract(1,np.divide(np.reshape(main_image, (1,784)),255))
            #plt.imshow(np.reshape(main_image_array[0],(28,28)))
            #plt.show()
            d1_images.append(main_image_array[0])

        coord.request_stop()
        coord.join(threads)

    return np.asarray(d1_images)

training_image_data = get_image_as_2d_array(training_image_queue)
testing_image_data = get_image_as_2d_array(testing_image_queue)
validation_image_data = get_image_as_2d_array(validation_image_queue)

print(training_image_data.shape)

print(training_image_data[0])

# We now need to create the index matching.

def create_index(training_image_data,training_image_class_queue):
    index_main = []
    size4 = len(training_image_data)
    for j in range(0, size4):
        size2 = len(d)
        index_temp = []
        for i in range(size2):
            index_temp.append(0)
        index_temp[training_image_class_queue[j]] = 1
        index_main.append(index_temp)

    index = np.asarray(index_main)
    return index

index_training = create_index(training_image_data,training_image_class_queue)
index_testing = create_index(testing_image_data,testing_image_class_queue)
index_validation = create_index(validation_image_data,validation_image_class_queue)
#Now the CNN :)

def get_next_batch_data(amount,times,i,training_image_data):
    each_batch = int(amount/times)
    j=each_batch*i
    next_batch = each_batch*(i+1)
    guba=[]
    while(j<next_batch):
        guba.append(training_image_data[j])
        j=j+1
    return(np.asarray(guba))

def get_next_batch_class(amount,times,i,index_training):
    each_batch = int(amount/times)
    j=each_batch*i
    next_batch = each_batch*(i+1)
    guba=[]
    while(j<next_batch):
        guba.append(index_training[j])
        j=j+1
    return(np.asarray(guba))

# Declaring the weights.
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
y_ = tf.placeholder(tf.float32, shape=[None, len(d)])

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

W_fc2 = weight_variable([1024,len(d)])
b_fc2 = bias_variable([len(d)])

y_conv = tf.matmul(h_fc1_drop , W_fc2)+ b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
'''
amount = len(training_image_data)
times = 1000
for i in range(times):
    input_data = get_next_batch_data(amount,times,i,training_image_data)
    input_index = get_next_batch_class(amount,times,i,index_training)
    train_step.run(session=sess,feed_dict={x: input_data, y_: input_index, keep_prob: 0.5})
    if i%100:
        input_data_testing = get_next_batch_data(amount,times,i,testing_image_data)
        input_index_testing = get_next_batch_class(amount,times,i,index_testing)
        print("test accuracy %g" % accuracy.eval(session=sess,feed_dict={x: input_data_testing, y_: input_index_testing, keep_prob: 1.0}))
'''
train_step.run(session=sess,feed_dict={x: training_image_data, y_: index_training ,keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: testing_image_data, y_: index_testing, keep_prob: 1.0}))
