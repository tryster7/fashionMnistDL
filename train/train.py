import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import pathlib
import sys
import json
import pandas as pd
import os

from sklearn.metrics import confusion_matrix

from google.cloud import storage
from keras.datasets import fashion_mnist


# Helper libraries
import numpy as np

def train(bucket_name, epochs=10, batch_size=128 ):

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
    
    predictions = model.predict(testX)
    
    pred = np.argmax(predictions, axis=1)
    
    #df = pd.DataFrame({'target': testy, 'predicted': pred}, columns=['target', 'predicted'])

    #df = df.applymap(np.int64)
    
    test_loss, test_acc = model.evaluate(testX,  testy, verbose=2)

    print("\n Test Accuracy is {} ".format(test_acc))

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

    #df_cm = df.groupby(['target', 'predicted']).size().reset_index(name='count')
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
    if len(sys.argv) < 1:
       print("Usage: train bucket-name epochs batch-size")
       sys.exit(-1)
    bucket_name = sys.argv[1]
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    train(bucket_name, epochs)

