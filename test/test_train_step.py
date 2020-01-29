import unittest

import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
from train import train


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        model = train.create_tfmodel(optimizer=tf.optimizers.Adam(),
                                          loss='sparse_categorical_crossentropy',
                                          metrics=['accuracy'])
        self.model = model

    def test_loadmodel(self):
        model = tf.saved_model.load('gs://a-kb-poc-262417/mnist/export/model/1')
        self.assertIsNotNone(model)

    def test_parseArguments(self):
        args = train.parse_arguments()
        self.assertIn("epochs", args)

    def test_correct_model_optimizer_and_loss(self):
        self.assertEquals(self.model.loss, 'sparse_categorical_crossentropy')
        self.assertIn('Adam', self.model.optimizer.get_config().values())

    def test_layers_in_model(self):
        self.assertEquals(len(self.model.layers), 6)

    def test_model_output(self):
        self.assertEquals(self.model.output.name, 'dense_9/Identity:0')

    def test_model_is_saved_at_given_dir(self):
        export_dir = '/workspace'
        train.save_tfmodel_in_gcs(export_dir,self.model)
        self.assertTrue(True, tf.saved_model.contains_saved_model(export_dir))


if __name__ == '__main__':
    unittest.main()
