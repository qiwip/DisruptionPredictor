import os
import sys
import tensorflow as tf
from CreateDataFrame import Cutter
import matplotlib.pyplot as plt


shot = int(sys.argv[1])
cutter = Cutter(normalized=True)
x, y = cutter.get_one(shot)

dataset = tf.data.Dataset.from_tensor_slices((x, y))
model = tf.keras.models.load_model(os.path.join('model', 'main', 'model.h5'))
model.evaluate(dataset)

print(model.summary())
y_ = model.predict(x)

plt.figure()
plt.plot(y, label='y')
plt.plot(y_, label='y_predict')
plt.legend()
plt.savefig(os.path.join('model', 'main', 'result_{}.png'.format(shot)))
