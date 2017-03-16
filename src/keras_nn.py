from itertools import cycle

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
DEF = tf.app.flags

DEF.DEFINE_integer('batch_size', 100, 'number of documents per batch')
DEF.DEFINE_integer('nb_epoch', 1, 'number of epochs')

DEF.DEFINE_integer('d1_size', 32, 'number of neurons in first dense layer')

def main():
    X = np.random.uniform(size=(1000, 3, 256, 256))
    y = np.random.choice(np.arange(10), size=1000)

    X = X.reshape((1000, -1))

    lb = LabelBinarizer().fit(y)

    X_train, y_train = X[:900], lb.transform(y[:900])
    X_test, y_test = X[900:], y[900:]


    def batch_generator(X, y=None):
        # TODO: shuffle the data each epoch
        # i either recommending just shuffling the order your batches are in
        # so that your cache is completely coherenent
        # OR doing a two-pass shuffle where shuffle in chunks of 1000
        #  but yield batches
        # a complete "proper" shuffle each epoch will destroy your cache for larger problems
        for start in cycle(range(0, len(X), FLAGS.batch_size)):
            end = start + FLAGS.batch_size
            if y is not None:
                yield X[start:end], y[start:end]
            else:
                yield X[start:end]

    model = Sequential([
        Dense(FLAGS.d1_size, activation='tanh', input_shape=X.shape[1:]),
        Dense(lb.classes_.shape[0], activation='softmax')
    ])

    model.compile(optimizer=SGD(), loss='categorical_crossentropy')
    model.fit_generator(batch_generator(X_train, y_train), nb_epoch=FLAGS.nb_epoch,
                        samples_per_epoch=X_train.shape[0])
    log_scores = model.predict_generator(batch_generator(X_test), X_test.shape[0])
    y_pred = lb.inverse_transform(log_scores)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()