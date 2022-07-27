import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical


# function to help extract data from the csv files in train_test_data
def get_csv_data(path):
    dataframe = pd.read_csv(path)

    labels = dataframe['label'].values
    labels_categorical = to_categorical(labels)
    dataframe.drop('label', axis=1, inplace=True)

    images = dataframe.values
    images = images / 255
    images = np.array([np.reshape(i, (28, 28, 1)) for i in images])

    return images, labels, labels_categorical

# prepare data
x_train, y_train, train_labels_catg = get_csv_data('../train_test_data/sign_mnist_train.csv')
x_test, y_test, test_labels_categorical = get_csv_data('../train_test_data/sign_mnist_test.csv')

x_train = np.expand_dims(x_train, axis=-1)  # it was 28,28, which cannot be inputted into a conv network,
x_test = np.expand_dims(x_test, axis=-1)  # So I used numpy expand dimension, which expands it to 1,28,28,1

# Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(26, activation='softmax'))

model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# fit, evaluate and save the model as a h5 file in this directory
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))
model.save('cnn.h5')

