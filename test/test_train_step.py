import unittest

import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
from train import train


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
        self.testy = testy
        self.trainX = trainX / 255
        self.testX = testX / 255
        self.trainX = self.trainX.reshape(trainX.shape[0], 28, 28, 1)
        self.testX = self.testX.reshape(testX.shape[0], 28, 28, 1)
        self.trainY = trainy

    def test_loadmodel_and_predict(self):
        model = tf.saved_model.load('gs://a-kb-poc-262417/mnist/export/model/1')
       
        #predictions = model.predict(self.trainX[0])
        #label = np.argmax(predictions, axis=1)
        #print(label)
        self.assertIsNotNone(model)

    def test_parseArguments(self):
        args = train.parse_arguments()
        self.assertIn("epochs", args)


if __name__ == '__main__':
    unittest.main()
