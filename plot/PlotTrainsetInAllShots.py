import os
import matplotlib.pyplot as plt


shots = list()
with open(os.path.join('..', 'log', 'ShotsInDataset.txt'), 'r') as f:
    for i in f.readlines():
        shots.append(int(i.split(' ')[0]))

train = list()
with open(os.path.join('..', 'log', 'ShotsUsed4Training.txt'), 'r') as f:
    for i in f.readlines():
        train.append(int(i))

plt.figure(figsize=(19.20, 10.80))
for i in shots:
    if i in train:
        plt.scatter(i, 0.5, s=2, marker='s', c='#196127')
    else:
        plt.scatter(i, 0.4, s=2, marker='s', c='#c6e48b')

plt.ylim(0, 1)
plt.yticks([0, 1], [0, 1])
plt.show()
