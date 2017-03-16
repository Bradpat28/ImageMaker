from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from PIL import Image
import numpy as np
import glob as glob
from tqdm import tqdm
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from itertools import cycle
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
"""
for file in glob.glob("images/startImages/*.jpeg"):
	img = Image.open(file)
	img.save(file[:-5] + ".jpg")
"""

batch_size = 16

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), dim_ordering="tf"))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

model.add(Conv2D(32, (3, 3), dim_ordering="tf"))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

model.add(Conv2D(64, (3, 3), dim_ordering="tf"))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('tanh'))

model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        'images/startImages/train',  # this is the target directory
        target_size=(100, 100),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'images/startImages/validation',
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='categorical')


model.fit_generator(
    train_generator,
    steps_per_epoch=129// batch_size,
    epochs=50,
    validation_data=train_generator,
    validation_steps=129// batch_size)

#model.save_weights('first_try.h5') 





