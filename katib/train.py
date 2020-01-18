import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import pathlib
import sys
import json
import os
import argparse

from keras.datasets import fashion_mnist


# Helper libraries
import numpy as np

def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('--epochs',
                      type=int,
                      default=5,
                      help='Number of epochs for training the model')
  parser.add_argument('--batch_size',
                      type=int,
                      default=64,
                      help='the batch size for each epoch')

  args = parser.parse_known_args()[0]
  return args

def train(epochs=10, batch_size=128 ):

    # load dataset
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

    #Data Normalization - Dividing by 255 as the maximum possible value 

    trainX = trainX / 255

    testX = testX / 255
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (28, 28, 1)))
    cnn.add(tf.keras.layers.MaxPooling2D(2,2))

    cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(64, activation='relu'))

    cnn.add(tf.keras.layers.Dense(10, activation = 'softmax'))

    cnn.summary()

    cnn.compile(optimizer=tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])

    cnn.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)
    
    test_loss, test_acc = cnn.evaluate(testX,  testy, verbose=2)
    print("Validation-accuracy={:.2f}".format(test_acc))
    print("test-loss={:.2f}".format(test_loss))

if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    args = parse_arguments()
    print(args)
    train(int(args.epochs), int(args.batch_size))
