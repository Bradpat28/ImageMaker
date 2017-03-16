import tensorflow as tf
from PIL import Image
import numpy as np
import glob as glob
from tqdm import tqdm
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from itertools import cycle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix




def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(label)
    return filenames, labels

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label

def write_data_files():
	writer = tf.python_io.TFRecordWriter("images.data")
	for x in tqdm(glob.glob("./images/*.jpg")):
		img = Image.open(x).convert("L")
		features = numpy.array(img)
		label = 1
		example = tf.train.Example(
			features = tf.train.Features(
				feature={
				'label' : tf.train.Feature(
					int64_list=tf.train.Int64List(value=[label])),
				'image' : tf.train.Feature(
					int64_list=tf.train.Int64List(value={features.astype("int64")})),
		}))
		serialized = example.SerializetToString()
		writer.write(serialized)

x = tf.placeholder(tf.float32, [None, 313 * 468])
W = tf.Variable(tf.zeros([313 * 468, 10]))
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

images = []
#read in the images
for x in glob.glob("./images/*.jpg"):
	img = Image.open(x).convert("L")
	img = np.array(img)
	images.append(img)
images = np.array(images)
y = np.random.choice(np.arange(10), size=2)

images.reshape((2, -1))
print(images)
lb = LabelBinarizer().fit(y)

X_train, y_train = images[:2], lb.transform(y[:2])
X_test, y_test = images[:2], y[:2]

def batch_generator(X, y=None):
	for start in cycle(range(0, len(X), 2)):
		end = start + 2
		if y is not None:
			yield X[start:end], y[start:end]
		else:
			yield X[start:end]

model = Sequential([
	Dense(32, activation='tanh', input_shape=images.shape[1:]),
    Dense(lb.classes_.shape[0], activation='softmax')
])

model.compile(optimizer=SGD(), loss='categorical_crossentropy')
model.fit_generator(batch_generator(X_train, y_train), nb_epoch=1,
                    samples_per_epoch=X_train.shape[0])

log_scores = model.predict_generator(batch_generator(X_test), X_test.shape[0])
y_pred = lb.inverse_transform(log_scores)
 

#write_data_files()

"""
images =[]

for x in glob.glob("./images/*.jpg"):
	#print(x)
	img = Image.open(x).convert("L")
	arr = numpy.array(img)
	images.append(arr)
print(images)

#images2 = ops.convert_to_tensor(images, dtype=dtypes.array)
labels = numpy.array([[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0]])

sess.run(train_step, feed_dict={x: images, y_: labels})

image_list, label_list = read_labeled_image_list("test.txt")
images = []

for x in image_list:
	img = Image.open(x).convert("L")
	arr = numpy.array(img)
	print (arr)
	images.append(arr)

numpy.savez("./hello", images=images, label_list=label_list)

l = numpy.load("./hello.npz")
trainimg_loaded = l['images']
trainlabel_loaded = l['label_list']


sess.run(train_step, feed_dict={x: trainimg_loaded, y_: trainlabel_loaded})

image_list, label_list = read_labeled_image_list("test.txt")
images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
labels = ops.convert_to_tensor(label_list, dtype=dtypes.float32)

input_queue = tf.train.slice_input_producer([images, labels],
                                        num_epochs=None,
                                        shuffle=True)

image, label = read_images_from_disk(input_queue)

image_batch, label_batch = tf.train.batch([image, label],
                                      batch_size=3, dynamic_pad=True)

sess.run(train_step, feed_dict={x: image_batch, y_: label_batch})
"""
