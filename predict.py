import os
import sys
import tensorflow as tf
import numpy as np
from CreateDataFrame import Cutter
import matplotlib.pyplot as plt

shots = list()
if len(sys.argv) > 2:
    shots = [int(sys.argv[1])]
else:
    with open(os.path.join('log', 'IsDisruptShots.txt'), 'r') as f:
        for i in f.readlines():
            shots.append(int(i.split(' ')[0]))
    with open(os.path.join('log', 'UnDisruptShots.txt'), 'r') as f:
        for i in f.readlines():
            shots.append(int(i.split(' ')[0]))

model = tf.keras.models.load_model(os.path.join('model', 'main', 'model.h5'))
print(model.summary())

cutter = Cutter(normalized=True)

path = os.path.join('model', 'main', 'result')
if not os.path.exists(path):
    os.makedirs(path)

for shot in shots:
    try:
        x, y = cutter.get_one(shot)
        y_ = model.predict(x)
        result = np.array([y, y_])
        np.save(os.path.join(path, 'y_y_{}.npy'.format(shot)), result)
    except Exception as e:
        print(shot, e)

# plt.figure()
# plt.plot(y, label='y')
# plt.plot(y_, label='y_predict')
# plt.legend()
# plt.savefig(os.path.join('model', 'main', 'result_{}.png'.format(shot)))
