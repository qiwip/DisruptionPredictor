import os
import pandas as pd
import tensorflow as tf
from CreateDataFrame import Cutter
import matplotlib.pyplot as plt


cutter = Cutter(normalized=True)
x, y = cutter.get_one(1065342)

model = tf.keras.models.load_model(os.path.join('model', 'main', 'model.h5'))

print(model.summary())
y_ = model.predict()

plt.figure()
plt.plot(y)
plt.plot(y_)
plt.savefig('result.png')
