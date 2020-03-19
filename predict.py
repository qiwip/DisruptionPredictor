import os
import sys
import tensorflow as tf
import numpy as np
from CreateDataFrame import Cutter


shots = list()
if len(sys.argv) > 1:
    shots = [int(sys.argv[1])]
else:
    with open(os.path.join('log', 'ShotsInDataset.txt'), 'r') as f:
        for i in f.readlines():
            shots.append(int(i.split(' ')[0]))

print(len(shots))

model = tf.keras.models.load_model(os.path.join('model', 'main', 'model.h5'))
print(model.summary())

cutter = Cutter(normalized=True)

path = os.path.join('model', 'main', 'result')
if not os.path.exists(path):
    os.makedirs(path)

for shot in shots:
    try:
        print(shot)
        x, y = cutter.get_one(shot)
        y_ = model.predict(x)
        y_ = np.array(y_).flatten()

        result = np.array([y, y_])
        np.save(os.path.join(path, 'y_y_{}.npy'.format(shot)), result)
    except Exception as e:
        print(shot, e)
