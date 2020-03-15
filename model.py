import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from DataSet import DataSet


BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000
learning_rate = 0.001
EPOCHS = 10


ds = DataSet()
train_dataset, test_dataset = ds.load()

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(13, 100)),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(learning_rate), metrics=['mae', 'mse'])

print(model.summary())


history = model.fit(train_dataset, epochs=EPOCHS, validation_split=0.2, verbose=0)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    # plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()
    plt.savefig('./mae.png')
    plt.close()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    # plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()
    plt.savefig('./mse.png')
    plt.close()


# plot_history(history)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

model.evaluate(test_dataset)

path = './model/model_{}.h5'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
# if not os.path.exists(path):
#     os.makedirs(path)

model.save(path)
