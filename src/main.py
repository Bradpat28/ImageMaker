import tensorflow as tf
import numpy as np
import glob as glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

x = tf.placeholder(tf.float32, [None, 100 * 100])
W = tf.Variable(tf.zeros([100 * 100, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer();

# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(init)

fileList = glob.glob('images/*.jpg')

image_list = []

for x in fileList:
    image = tf.read_file(x)
    image_tensor = tf.image.decode_jpeg(x, channels=3)
    image_list.append(image_tensor)

for i in range(1000):
    batch = tf.train.batch(image_list, batch_size=2, enqueue_many=True, capacity=2, dynamic_pad=True)
    sess.run(train_step, feed_dict={x: })
