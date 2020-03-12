import tensorflow as tf
from DataSet import DataSet


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
learning_rate = 0.001

# ds = DataSet()
# train_dataset, test_dataset = ds.load()
#
# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4, 100)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# model.fit(train_dataset, epochs=10)
