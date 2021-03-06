import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import pathlib

import sys
import json
import pandas as pd
import os
import argparse
import csv

from sklearn.metrics import confusion_matrix

from google.cloud import storage
from keras.datasets import fashion_mnist

from kubeflow.metadata import metadata
from datetime import datetime
from uuid import uuid4

# Helper libraries
import numpy as np

METADATA_STORE_HOST = "metadata-grpc-service.kubeflow"  # default DNS of Kubeflow Metadata gRPC serivce.
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


def train(bucket_name, epochs=2, batch_size=512):

    exec = create_metadata_execution()
    testX, testy, trainX, trainy = load_and_normalize_data()
    cnn = create_tfmodel(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn.summary()

    cnn.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)

    predictions = cnn.predict(testX)

    pred = np.argmax(predictions, axis=1)

    test_loss, test_acc = cnn.evaluate(testX, testy, verbose=2)

    print("\n Test Accuracy is {} ".format(test_acc))
    print("\n Test Loss is {} ".format(test_loss))

    save_tfmodel_in_gcs(exec, bucket_name, cnn, batch_size, epochs, test_loss, test_acc)

    df = pd.DataFrame({'target': testy, 'predicted': pred}, columns=['target', 'predicted'])
    df = df.applymap(np.int64)

    create_kf_visualization(bucket_name, df, test_acc)


def save_tfmodel_in_gcs(exec, bucket_name, model, batch_size, epochs, test_loss, test_acc):

    export_path = bucket_name + '/export/model/1'
    tf.saved_model.save(model, export_dir=export_path)
    model_metadata = save_model_metadata(exec, batch_size, epochs, export_path)
    save_metric_metadata(exec, model_metadata, test_acc, test_loss, bucket_name)



def create_tfmodel(optimizer, loss, metrics):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(tf.keras.layers.MaxPooling2D(2, 2))
    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(64, activation='relu'))
    cnn.add(tf.keras.layers.Dense(10, activation='softmax'))
    cnn.compile(optimizer, loss, metrics)
    return cnn

  
def create_kf_visualization(bucket_name, df, test_acc):
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': str(test_acc),
            'format': "PERCENTAGE"
        }]
    }

    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

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

    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


def save_metric_metadata(exec, model, test_acc, test_loss, bucket_name):

    metric_file = bucket_name + '/metadata/metrics.csv'

    with  file_io.FileIO(metric_file, 'w') as f: 
        metric_writer = csv.writer(f)
        metric_writer.writerow(['accuracy', test_acc])
        metric_writer.writerow(['loss',  test_loss])

    # Save evaluation
    metrics = exec.log_output(
        metadata.Metrics(
            name="MNIST-evaluation",
            description="validating the MNIST model to recognize images",
            owner="demo@kubeflow.org",
            uri = metric_file,
            model_id=str(model.id),
            metrics_type=metadata.Metrics.VALIDATION,
            values={"accuracy": str(test_acc),
                    "test_loss": str(test_loss)},
            labels={"mylabel": "l1"}))
    print("Metrics id is %s" % metrics.id)


def save_model_metadata(exec, batch_size, epochs, export_path):

    training_file = 'gs://dlaas-model/metadata/model.csv'

    with  file_io.FileIO(training_file, 'w') as f:
        metric_writer = csv.writer(f)
        metric_writer.writerow(['model_framework', 'tensorflow', 'v2.0'])
        metric_writer.writerow(['learning_rate', 0.5])
        metric_writer.writerow(['epoch', epochs ])
        metric_writer.writerow(['batch_size', batch_size ])
        metric_writer.writerow(['layers',"28, 28, 1" ])

    # Save model;
    model_version = "model_version_" + str(uuid4())
    model = exec.log_output(
        metadata.Model(
            name="MNIST",
            description="model to recognize images",
            owner="demo@kubeflow.org",
            uri=export_path,
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
    return model


def create_metadata_execution():
    global metadata
    # Create Metadata Workspace and a Exec to log details
    mnist_train_workspace = metadata.Workspace(
        # Connect to metadata service in namespace kubeflow in k8s cluster.
        store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
        name="mnist train workspace",
        description="a workspace for training mnist",
        labels={"n1": "v1"})
    run1 = metadata.Run(
        workspace=mnist_train_workspace,
        name="run-" + datetime.utcnow().isoformat("T"),
        description="a run in ws_1")
    exec = metadata.Execution(
        name="execution" + datetime.utcnow().isoformat("T"),
        workspace=mnist_train_workspace,
        run=run1,
        description="execution example")
    print("An execution was created with id %s" % exec.id)
    return exec


def load_and_normalize_data():
    # load dataset
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
    # Data Normalization - Dividing by 255 as the maximum possible value
    trainX = trainX / 255
    testX = testX / 255
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

    return testX, testy, trainX, trainy


if __name__ == '__main__':

    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size")
        sys.exit(-1)

    args = parse_arguments()
    print(args)
    train(args.bucket_name, int(args.epochs), int(args.batch_size))
