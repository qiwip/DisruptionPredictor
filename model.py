import os
import time
import tensorflow as tf
from DataSet import DataSet


BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000
learning_rate = 0.001

ds = DataSet()
train_dataset, test_dataset = ds.load()

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(13, 100)),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_dataset, epochs=50)

model.evaluate(test_dataset)
path = './model/model_{}'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
if not os.path.exists(path):
    os.makedirs(path)

model.save(path)
