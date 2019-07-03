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

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset,validation_steps=3)

result = model.predict(data, batch_size=32)
print(result.shape)