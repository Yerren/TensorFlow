from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels

inputs = tf.keras.Input(shape=(32,))

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)


# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))


# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)
