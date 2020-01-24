import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import pathlib

import sys
import json
import pandas as pd
import os
import argparse

from sklearn.metrics import confusion_matrix

from google.cloud import storage
from keras.datasets import fashion_mnist

from kubeflow.metadata import metadata
from datetime import datetime
from uuid import uuid4

# Helper libraries
import numpy as np


METADATA_STORE_HOST = "metadata-grpc-service.kubeflow" # default DNS of Kubeflow Metadata gRPC serivce.
METADATA_STORE_PORT = 8080

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--bucket_name',
                      type=str,
                      default='gs://',
                      help='The bucket where the output has to be stored')
  parser.add_argument('--epochs',
                      type=int,
                      default=1,
                      help='Number of epochs for training the model')
  parser.add_argument('--batch_size',
                      type=int,
                      default=64,
                      help='the batch size for each epoch')

  args = parser.parse_known_args()[0]
  return args

def train(bucket_name, epochs=10, batch_size=128 ):

    global metadata
    #Create Metadata Workspace and a Exec to log details
    mnist_train_workspace = metadata.Workspace(
    # Connect to metadata service in namespace kubeflow in k8s cluster.
    store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
    name="mnist train workspace",
    description="a workspace for training mnist",
    labels={"n1": "v1"})

    run1 = metadata.Run(
    workspace=mnist_train_workspace,
    name="run-" + datetime.utcnow().isoformat("T") ,
    description="a run in ws_1")

    exec = metadata.Execution(
    name = "execution" + datetime.utcnow().isoformat("T") ,
    workspace=mnist_train_workspace,
    run=run1,
    description="execution example")

    print("An execution was created with id %s" % exec.id)

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
    
    predictions = cnn.predict(testX)
    
    pred = np.argmax(predictions, axis=1)
    
    df = pd.DataFrame({'target': testy, 'predicted': pred}, columns=['target', 'predicted'])

    df = df.applymap(np.int64)

    #Save model;
    model_version = "model_version_" + str(uuid4())
    model = exec.log_output(
        metadata.Model(
                name="MNIST",
                description="model to recognize images",
                owner="demo@kubeflow.org",
                uri="gs://a-kb-poc-262417/mnist/export/model",
                model_type="CNN",
                training_framework={
                    "name": "tensorflow",
                    "version": "v2.0"
                },
                hyperparameters={
                    "learning_rate": 0.5,
                    "layers": [28, 28, 1],
                    "epochs": str(epochs),
                    "batch-size": str(batch_size),
                    "early_stop": True
                },
                version=model_version,
                labels={"tag": "train"}))
    print(model)
    print("\nModel id is {0.id} and version is {0.version}".format(model))

    test_loss, test_acc = cnn.evaluate(testX,  testy, verbose=2)

    print("\n Test Accuracy is {} ".format(test_acc))
    print("\n Test Loss is {} ".format(test_loss))
    #Save evaluation
    metrics = exec.log_output(
    metadata.Metrics(
            name="MNIST-evaluation",
            description="validating the MNIST model to recognize images",
            owner="demo@kubeflow.org",
            uri="gs://a-kb-poc-262417/mnist/metadata/mnist-metric.csv",
            model_id=str(model.id),
            metrics_type=metadata.Metrics.VALIDATION,
            values={"accuracy": str(test_acc),
                    "test_loss": str(test_loss)},
            labels={"mylabel": "l1"}))

    print("Metrics id is %s" % metrics.id)

    export_path = bucket_name + '/export/model/1' 

    tf.saved_model.save(cnn, export_dir=export_path)

    metrics = {
    	'metrics': [{
            'name': 'accuracy-score',
            'numberValue': str(test_acc),
            'format': "PERCENTAGE"
        }]
    }

    vocab = list(df['target'].unique())

    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])

    cm_file = bucket_name + '/metadata/cm.csv'

    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)


    print("***************************************")
    print("Writing the confusion matrix to ", cm_file)

    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            'labels': list(map(str, vocab)),
        }]
    }


    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
       print("Usage: train bucket-name epochs batch-size")
       sys.exit(-1)

    args = parse_arguments()
    print(args)
    train(args.bucket_name, int(args.epochs), int(args.batch_size))

