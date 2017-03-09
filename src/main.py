import tensorflow as tf
import numpy as np

image = tf.image.decode_jpeg("OPicture1.jpg");
resized_image = tf.image.resize_images(image, [300, 300]);



